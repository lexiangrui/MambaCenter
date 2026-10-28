# CP-Mamba 实现说明

设计文档：[CP-Mamba：中心先验引导的选择性扫描用于小目标检测](https://claude.ai/artifact/Q3i5UKMEjNnE76rRzjjkfh)。
代码：[mambacenter/cp_mamba.py](../mambacenter/cp_mamba.py)，配置：[configs/cpmamba.yaml](../configs/cpmamba.yaml)（B5）、[configs/cpmamba_b1.yaml](../configs/cpmamba_b1.yaml)（B1）。

## 结构

DLA-34（CenterNet ImageNet 权重）→ Mamba-FPN：P5/P4/P3 各 2 个普通 VSS 块（四向），P2 为 1 个 CP-SSM 块（expand=1）→ CenterNet 热图 / 宽高 / 偏移头（stride 4）。

| 设计文档 | 实现 |
|---|---|
| 3.2 粗热图 H_c | `CoarseHeatmap`：P3 输出上 Conv3×3(128)+ReLU+Conv1×1(10 类)，偏置 −2.19；输出 `hm_c` |
| 3.2 先验 S | `centre_prior`：max_c σ(H_c) → 双线性 ×2 → 3×3 最大池化，s∈[0,1] |
| 3.2 粗热图监督 | CenterNet 数据管线生成 stride-8 高斯目标 `hm_c`（同一套框、同一半径公式），focal loss |
| 3.3 Δ 调制 | Δ = softplus(W_Δx + b_Δ + α s)，作用于全部路由；α 用 `nonneg`（值≥0、梯度直通），零初始化 |
| 3.4 热图优先顺序 | `priority_order` / `select_tokens`：s 量化为 B=4 档，低→高，档内光栅序（稳定排序） |
| 3.4 与常规方向并用 | 水平、垂直两条稠密路由 + 1 条优先路由，输出相加 |
| 3.5 稀疏扫描 | 优先路由只扫描 top-ρN（ρ=15%）；训练时另加 5% 随机 token；未选中 token 只走块残差 |
| 3.6 损失 | L_hm + 0.1 L_wh + L_off + λ_c L_hm^coarse，λ_c=0.5；CP-SSM 梯度默认回传到粗热图分支 |
| 5.1 warm-up | 前 5 轮关闭优先路由，之后 5 轮线性淡入（`prio_gate`，随 checkpoint 保存） |
| 3.7 实现 | argsort / gather / scatter，原生 selective_scan_cuda 不改；批内 K 相同 |

## 外挂版本（第二批）

代码：[mambacenter/addon.py](../mambacenter/addon.py)，配置：[configs/addon_cp.yaml](../configs/addon_cp.yaml)（B5′）、[configs/addon_cp_b1.yaml](../configs/addon_cp_b1.yaml)（B1′），对照为 [configs/addon_vss.yaml](../configs/addon_vss.yaml)（A1′）。

原版 CenterNet DLA-34 + DLAUp/IDAUp（16 个 DCN）不变，只在两处插入残差块：

- DLAUp 的 P3（stride 8，128 通道）上 2 个普通 VSS 块，之后接粗热图头（结构同上表），再进入最后的 IDAUp；
- IDAUp 输出（stride 4，64 通道）上 2 个 CP-SSM 块，之后进入检测头。

每个块都带 FFN（mlp_ratio 4）。混合器和 FFN 的末端投影零初始化，所以初始化时检测器的输出与 DCN 基线相同。

相对第一批有两处改动：

| 项 | 第一批 | 外挂版本 | 原因 |
|---|---|---|---|
| α 初值 | 0 | 1.0（`alpha_init`），先验峰值处 Δ 约放大 e ≈ 2.7 倍 | 第一批 α 从 0 只学到 0.12（Δ 最多放大约 12%），第 90 轮后不再变化 |
| α 生效时间 | 从第 0 轮起 | 与优先路由共用 warm-up 和淡入（`ramp_delta`） | 粗热图收敛前不调制 Δ，对应设计文档 5.1 的 warm-up |

外挂新增的参数（各个块和粗热图头）用共享 lr 的 5 倍（`train.new_lr_mult`）训练。

## 消融开关（`model.cp`）

| ID | 设置 |
|---|---|
| B1 | `use_cp_block: false`（P2 为普通 VSS，粗热图只作辅助监督） |
| B2 / B3 / B4 | `delta_mod` / `priority` / `sparse_ratio` 单独开启 |
| B6 | `descending: true` |
| B7 | `bins: 1 / 2 / 4 / 8 / null`（null = 完全排序） |
| B8 | `sparse_ratio: 0.05 … 1.0` |
| B9 | `extra_ratio: 0` |
| B10 | `dilate: false` |
| B12 | `detach_prior: true` |

B11（P4 粗热图、无监督显著性、随机先验）尚未实现。B2–B4 中常规路由数量（两向或四向）需在启动前确定，当前默认两向。
