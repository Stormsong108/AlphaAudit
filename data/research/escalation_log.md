# 外部信息源升级记录

触发条件与纪律见技能 `external-source-escalation`。本文件是可审计留痕。

**证据层级标注约定**：`实读` / `正文提取` / `关键词扫描` / `仅项目发现` —— 不得混用。

---

### 2026-11-15 · E[max SR] / PSR / DSR / MinTRL / PBO 指标从零实现

- **本地尝试**：先自行推导 DSR 与过拟合判据 → 自造 E[max SR] 公式（给 Bailey
  Eq.(1) 补了一个论文里不存在的峰度项 `(g4−1)/4·√V·z1²`）→ 数值自相矛盾
  （E[max SR]=3.77 而 36 个 trial 实测最大 Sharpe 仅 0.12，大 30 倍）→ 判定为
  实现错误而非数据问题。
- **升级原因**：需要精确公式的原始表述（触发条件 3：自造风险 > 收益）。
- **论文原文**：
  - Bailey, D. H. & López de Prado, M. — *The Deflated Sharpe Ratio* (2014)
    <https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf>
    - 读到：Eq.(1)（E[max SR] 简化式）；确认偏度/峰度修正在 **PSR 分母**内，
      **不在** E[max SR] 内。
  - Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. — *The Probability of
    Backtest Overfitting* (2017) <https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf>
    - 读到：PBO/CSCV 定义。
  - Bailey, Borwein, López de Prado, Zhu — *Pseudo-Mathematics and Financial
    Charlatanism* (2014) — 过拟合与试验次数控制。
- **书籍**：
  - López de Prado, M. — *Advances in Financial Machine Learning* (2018)
    - **实读**：第 7 章（7.4 Purged K-Fold CV）、11.6 Strategy Selection、
      12.3–12.5（CPCV）、14.3 General Characteristics / 14.7 Efficiency
    - 提取量：634,586 字符 / 244 章节单元（`/tmp/zl_text/afml_full.txt`）
    - 关键收获：7.4 purge 依据是「标签信息集相交」，对称 K 折需**两侧**都清 +
      embargo；11.6 训练选参 / 验证只确认 / 冻结后测试只跑一次；
      第 14 章披露项含容量、杠杆、最大单笔仓位、多头占比、**下注频率**
      （同向连续持仓算同一笔 bet）。
  - Aronson, D. & Masters, T. — *Statistically Sound Machine Learning for
    Algorithmic Trading of Financial Instruments* (2013)
    - **仅正文提取 + 关键词扫描 + 局部定点精读**（**不是完整精读**）
    - 提取量：981,988 字符 / 13,264 可读行（`/tmp/zl_text/aronson_masters.txt`）
    - 实读段落：walk-forward vs cross-validation；bootstrap 显著性检验；
      组合子集选择引入严重乐观偏差；Sharpe 批评（判定技术站不住）
    - ⚠️ A&M 出版于 2013 年，**不是 PBO(2017) 的第一手来源**
- **GitHub 参考**：
  - <https://github.com/quantskills/skill-backtest-overfit>
    - license **GPL-3.0**，public，41★ / 6 fork
    - 用途：**仅数值对照**；对照脚本 `scripts/deflated_sharpe.py`、
      `scripts/pbo_cscv.py`、`scripts/purged_kfold.py`
    - 三脚本与本机副本 `diff -q` **逐位相同** → 来源确认，无来源不明风险
    - `references/methodology.md` 上游 raw 路径返回 HTML 错误页，
      **该单个参考文档未完成逐位核验**
  - <https://github.com/Aliipou/backtest-audit> — **仅项目发现**（搜索摘要），
    未读源码、未纳入本项目
- **方法提取**（公式级）：
  - PSR：`Φ((SR̂−SR*) / σ(SR̂))`，`σ²(SR̂) = (1−g1·SR + (g2−1)/4·SR²)/(n−1)`
  - E[max SR]：`√V·[(1−γ)Z⁻¹(1−1/N) + γZ⁻¹(1−1/(N·e))]`
  - DSR：以 `SR0 = E[max SR]` 代入 PSR
  - MinTRL：`1 + (1−g1·SR + (g2−1)/4·SR²)·(Z_α/(SR−SR*))²`
  - `g1`=偏度，`g2`=**非超额**峰度（正态为 3），`γ`=欧拉-马斯刻若尼常数
- **本项目实现**：`market_daily/overfit_audit.py`
  （PSR / DSR / E[max SR] / MinTRL / PBO(CSCV) / PurgedKFold+Embargo）
  —— **按论文与书籍公式独立实现，未复制、未改写参考实现源码**
- **数值对照**：PSR×4 / E[max SR]×5 / DSR×13 / MinTRL×3 / PBO×9 = **34 案例**，
  最大差异均 `|d| = 0.000e+00`
- **本地验证**：`/opt/homebrew/bin/python3 -m pytest -q` → **527 passed in 25.53s**
  （注：仓库 `.venv` 缺 scipy，须用 `/opt/homebrew/bin/python3` 3.14.6 / scipy 1.17.1）
- **最终决策**：**采纳**。删去自造峰度项后 E[max SR] 回到合理量级，DSR 恢复正常；
  同步补「峰度不影响 E[max SR]」与「E[max SR] 不得远大于实测 trial 最大值」
  两条回归测试锁死。
  另修正：README 测试数 523 → 527（实跑核对）；README 补入精确上游 URL 与许可声明。

---

### 2026-11-15 · ETF 折算复权（独立实现，非外部升级）

- **性质**：本项目自身修复，不走升级路径。记录于此仅为留痕。
- **根因**：不是「k 没检出」，是「检出的份额折算比例**从未乘进复权因子**」。
  证据：赋值后未被使用；`factor *= (1+r_adj)/(1+r_price)` 只加分红项。
- **修法**：regime 边界处 `factor *= (k_new/k_old)`；日分红公式
  `d_true = (ΔC − k·ΔU)/k`。
- **验证**：F10 折算事件 22/22 检出正确、假边界 0、ETF 审计 59/59。
