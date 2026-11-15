# market_daily — A股量化研究与日报系统

> **这是一个研究系统，不是交易系统。当前没有任何策略被放行，也没有任何实盘能力。**

一个面向 A 股 / ETF 的量化研究流水线：数据刷新 → 因子发现 → 样本外回测 → 过拟合审计 → 每日研究汇报。
核心取向是**先证伪自己**：每个结论都要过门禁，门禁不过就不放行。

---

## ⚠️ 免责声明

- 本项目**仅供研究与学习**，不构成投资建议。
- 所有回测均为模拟，**不具备真实下单能力**，也未接入任何券商交易接口。
- 回测收益不代表未来收益。历史表现与未来表现无必然联系。

---

## 当前结论：三条通道均不放行

| 通道 | 结果 | 卡在哪 |
|---|---|---|
| **ETF 轮动**（59 只 ETF，1971 期） | 测试段净年化 +15.52%，基准 +10.03%，**超额 +5.49%** | ❌ Sharpe 0.443 < 0.5（基准 0.624）；**PSR(超额 vs 0)=0.61**；MinTRL 需 1713 期 vs 实测 49 期 |
| **A股择时**（market timing） | `passes_bonferroni=false` | ❌ `long_flat` 相对基准约 **−3.20%/年** |
| **A股横截面 11 因子** | 多头净年化 +16.40%，等权基准 +18.43% | ❌ **净超额 −2.03%/年**（三段一致为负 = 结构性负 alpha） |

**ETF 通道的失败原因不是策略差，是样本不够**：PSR(vs 0)=0.61 意味着 49 期样本连"超额显著大于 0"都证不了。
这是本项目最有价值的结论 —— 它把"看起来赚钱"和"能被证明"分开了。

---

## 方法学要点

这些是项目里**踩过坑并已修正**的口径，写在这里是因为它们比策略本身更值得复用：

### 1. 训练/验证/测试三段严格隔离

```
训练段 ──────► 选参（唯一来源）
验证段 ──────► 只判通过/不通过，不参与选参
冻结件 ──────► 落盘 + spec_hash
测试段 ──────► 从磁盘读回校验后只跑一次
```

违反这条的典型后果：把验证段当第二个训练段（`chosen = passed[0]`），得到漂亮的验证段数字和崩掉的测试段。

### 2. 回测口径陷阱（全部有测试锁定）

| 陷阱 | 正确做法 |
|---|---|
| 年化用 `× 252` | 回测以「持有 N 日」为一期 → 年化用 `per_period × (252/hold)` |
| 策略与基准期次起点错开 | 两条时间网格必须对齐到同一批期次 |
| 基准相减 | 策略与基准必须**逐期**相减，不能拿年化数字相减 |
| Sharpe 年化因子除反 | `Sharpe = (mean/sd) × √ann`，ann = 252/期长 |
| 复权基准 | 东财后复权 `fqt=2`；前复权 `fqt=1` 会让长期高分红股历史价穿零 |

**抓 bug 的通用手法**：同一个量在系统里被几处独立算法算出来 → 交叉比对数值，不一致就是口径串了。
（本项目实测抓到：同一"净超额"在两处算差 5 倍；Sharpe 被除反 25 倍；E[max SR] 比所有观测值大 30 倍。）

### 3. 不可交易时段偏差（2026-11 最重要的发现）

11 因子反转信号的超额**几乎全在「信号日收盘 → 次日开盘」跳空段**（+2.13%/年），
而信号在 `close[t]` 才算出（15:00 定盘）、**最早只能 `open[t+1]` 成交** → 结构上吃不到；
**可交易的买入日盘中段反而是负的**（−2.87%/年）。

> 「按收盘价成交」的回测是**双向作弊**：白赚买不到的跳空 + 躲开开盘后的下跌。

### 4. 过拟合审计（`overfit_audit.py`）

独立实现 PSR / DSR / E[max SR] / MinTRL / PBO(CSCV) / PurgedKFold / 多重检验门槛，
并与公开参考实现做**逐位数值交叉验证**（PSR×4、E[max SR]×5、DSR×13、MinTRL×3、PBO×9 全部 `|d|=0.000e+00`）。

两种 purge 语义**严格分离**：

- `purged_train_bounds` —— 前向链式，训练样本整体早于测试段，只需清除尾部
- `PurgedKFold` —— 对称 K 折，训练集包含测试段前后样本，按**标签信息集相交**在两侧 purge，并施加 embargo

> ⚠️ PBO=0 **不能**单独当作"没过拟合"的证据。候选高度相关时（实测平均相关 0.7683），
> "样本内最优"几乎必然也是"样本外最优"，PBO 结构性偏低。本项目因此加了**判别力前置条件**：
> 候选平均相关 > 0.70 时，PBO 门禁直接判不通过 —— **无法判别 ≠ 通过**。

### 5. 成本是天花板

A股单笔 ≥2万 的**纯费率下限 = 10.2bp 往返**（佣金万2.5双边 + 印花税卖出5bp + 沪市过户费），
加滑点 → 20.2bp。策略的毛边际必须 ≥ 这个数才可能盈利。

实测：canonical 多空毛边际 5.24bp/期 —— **比成本下限小一个量级**。
这不是调参问题。

### 6. 幸存者偏差

上交所官方名单 2015-2026 窗口内退市 A 股 106 只，universe 只含 3 只 → **103 只被剔除**。
补回 87 只重算后，「多头跑不赢等权基准」的结论**修正前后都成立** → 偏差量级小、不阻断。
但本项目策略是**反转类**，暴跌股正是随后退市的那批 → 退市股缺失会**系统性高估**反转策略收益。

---

## 快速开始

```bash
python3 -m pytest tests/ -q        # 523 passed
python3 -m pytest tests/test_overfit_audit.py -q   # 过拟合审计专项
```

主要模块：

| 文件 | 作用 |
|---|---|
| `overfit_audit.py` | PSR / DSR / E[max SR] / MinTRL / PBO / PurgedKFold / 多重检验 |
| `etf_audit.py` | ETF 通道过拟合审计（接入门禁） |
| `etf_lab.py` / `etf_research.py` | ETF 面板、因子、回测、分段与冻结件 |
| `daily_loop.py` | 每日 refresh → predict → judge 闭环 |
| `render/report.py` | HTML 日报渲染 |
| `research_evidence.py` | G1–G4 门禁定义 |

---

## 数据源与第三方

- 行情：东方财富（主）、腾讯（回退）、新浪
- ETF 净值与折算记录：交易所 F10
- 退市名录：上交所官方终止上市清单接口

### 第三方致谢与来源声明

本项目的统计方法**依据公开论文与书籍公式独立实现**，未复制任何第三方代码：

- Bailey, D. H. & López de Prado, M. — *The Deflated Sharpe Ratio* (2014)
  公开 PDF：<https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf>
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. — *The Probability of Backtest Overfitting* (2017)
  公开 PDF：<https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf>
- Bailey, Borwein, López de Prado, Zhu — *Pseudo-Mathematics and Financial Charlatanism* (2014)
- López de Prado, M. — *Advances in Financial Machine Learning* (2018)
  第 7 章（purged CV / embargo）、第 12 章（CPCV）、第 14 章（回测统计）
- Aronson, D. & Masters, T. — *Statistically Sound Machine Learning for Algorithmic Trading of Financial Instruments* (2013)

> 📌 **来源归因更正**：A&M 出版于 **2013 年**，早于 Bailey/López de Prado 的 PBO 论文（2017）。
> 因此 A&M **不是** PBO 的第一手来源；PBO 的第一手来源是上列 2017 年公开论文。
> A&M 在本项目中的作用是交叉验证、Sharpe 比率与试验次数控制的教学参考。

数值交叉验证时使用过一个 GPL-3.0 许可的公开参考实现，**仅作数值对照**，
本仓库的代码与公式均按论文/书籍独立实现，未复制或改写其源码。

---

## 许可

代码：[MIT License](LICENSE)

数据与研究产物由脚本生成，不随代码授权。
