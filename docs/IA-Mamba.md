# IA-Mamba 实现说明

设计文档：[IA-Mamba：创新驱动状态空间模型用于小目标检测](https://claude.ai/artifact/VSQvLagDaDRPgiq7WnAcWf)。
代码：[mambacenter/ia_mamba.py](../mambacenter/ia_mamba.py)，配置：[configs/iamamba.yaml](../configs/iamamba.yaml)（A4）；A1 为 [configs/mambafpn.yaml](../configs/mambafpn.yaml)。

## 结构

DLA-34（CenterNet ImageNet 权重）→ IA-FPN：P5/P4/P3 各 2 个 IA-VSS 块（expand=2），P2 1 个（expand=1）→ CenterNet 头；热图头输入拼接 P2–P4 的创新图（3 通道）。

| 设计文档 | 实现 |
|---|---|
| 3.2 选择性 SSM | 原生 selective_scan_cuda；四向路由 →、↓、←、↑，投影与扫描保持 fp32 |
| 3.3 一步预测 | x̂_t = P y_{t−1}：扫描输出沿序列右移一位（y_{−1}=0），P 为 d×d 线性层（可选逐通道），零初始化 |
| 3.3 创新量与局部归一化 | e_t = x_t − x̂_t；s = ‖e‖₂ / (AvgPool_7(‖e‖₂) + ε)，在各路由还原到二维后计算 |
| 3.4 四向软最小 | E = −τ log Σ_d exp(−s_d/τ)，τ=0.5 |
| 3.5 有下界调制 | Δ^(l+1) = softplus(W_Δx + b) + γ softplus(aE^(l) + c)，以 delta_softplus=False 传入内核；γ≥0（`nonneg`）零初始化，a=1、c=0 |
| 同层跨块 | 第 l 块的 E 调制同一层第 l+1 块；P2 只有 1 块，按设计文档不受调制 |
| 3.6 创新图入热图头 | 各层最后一块的 E（P2、P3、P4）双线性上采样到 stride 4 与 P2 特征拼接；新增输入通道权重零初始化 |
| 3.7 损失 | L_hm + 0.1 L_wh + L_off + λ_pred·mean(sg(x_t) − x̂_t)²，λ_pred=0.1；按元素平均（各层通道数不同时尺度一致） |
| 3.8 开销 | 创新量计算在反向时重算（checkpoint），只多保存扫描输出 |

## 外挂版本（第二批）

代码：[mambacenter/addon.py](../mambacenter/addon.py)，配置：[configs/addon_ia.yaml](../configs/addon_ia.yaml)（A4′），A1′ 为 [configs/addon_vss.yaml](../configs/addon_vss.yaml)。

原版 CenterNet DLA-34 + DLAUp/IDAUp（16 个 DCN）不变。在 DLAUp 的 P3（stride 8，128 通道）和 IDAUp 输出（stride 4，64 通道）上各接 2 个 IA-VSS 残差块；每个块带 FFN（mlp_ratio 4），末端投影零初始化。

- **调制路径：** 设 `cross_level: true`。stride-4 的第一块读 stride-8 最后一块的 E，第二块读第一块的 E，所以两个 stride-4 块都受调制。第一批的 P2 只有 1 块，从未受过调制；γ 也只在 P3 第二块学到了 0.067。stride-8 的第一块前面没有 E，它的 γ、a、c 不参与计算。
- **热图头：** 两处最后一块的 E 先上采样到 stride 4，再经一个零初始化的 3×3 卷积，加到热图头第一层卷积的输出上。这等价于把 E 作为额外输入通道拼进去、对应权重零初始化。
- **初始化与学习率：** γ 仍零初始化。外挂新增的参数（含 P、γ、a、c 和上面这个卷积）用共享 lr 的 5 倍（`train.new_lr_mult`）训练。
- **A5′（[configs/addon_ia_a5.yaml](../configs/addon_ia_a5.yaml)）：** `lower_bound: false`，Δ = softplus(W_Δx + b + γ(aE + c))。γ 仍 ≥ 0，a、c 不受限制，所以 aE + c < 0 的位置 Δ 会被压低。加跑它是因为 A4′ 里 γ 被训练推成了负数、在前向时被截到 0。注意：A5′ 里如果 γ 早期同样变成负数，a、c 就收不到梯度，这一组会退化成不调制，需要看它的 γ 再下结论。

## 消融开关（`model.ia`）

| ID | 设置 |
|---|---|
| A1 | `configs/mambafpn.yaml`（普通 VSS 颈部） |
| A2 / A3 | `modulate: false` / `to_head: false` |
| A5 | `lower_bound: false` |
| A6 | `target: B`（只放大写入增益 B̄） |
| A7 | `fusion: mean / max / softmin` |
| A8 | `local_norm: false` |
| A9 | `loss.pred_weight: 0` |
| — | `cross_level: true`：让上一层末块的 E 调制下一层首块（P2 因此也受调制），设计文档之外的选项 |

A10（GT 热图监督 E）、A11（信号源替换）、A12（同层两遍扫描）尚未实现。
