# CEA 医学图像分类训练计划

## 使用规则

- 所有实验开始前必须先完整读取并遵守 `PROTOCOL.md`。
- 每个 Step 开始前必须读取本文件中对应的步骤，确认当前 Step 的目标、允许改变的变量和输出要求。
- 本文件不是一次性锁定的执行顺序；每个 Step 完成后必须根据实际结果审核后续步骤，后续步骤可以调整、暂停或停止。
- 任何后续 Step 的目标、输入组合、主要变量或输出要求发生变化，都必须在开始该 Step 前由用户确认并记录。
- 当前 Prompt 必须明确本次执行的 Step；未明确 Step 时，不得开始训练或评估。
- 除当前 Step 明确要求改变的变量外，其余训练设置保持不变。
- 如果本文件与 `PROTOCOL.md` 冲突，以 `PROTOCOL.md` 为准；必须先报告冲突，不得自行解释或覆盖协议。
- 每个 Step 开始前必须完成 `PROTOCOL.md` 要求的数据完整性检查；检查失败时立即停止。
- 每个正式 Benchmark Step 完成后保存原始 CSV / JSON、Mean ± Std 汇总和规定的柱状图结果，并记录实际配置。
- DeepLIFT / 通道扰动等解释性分析 Step 必须保存原始分析结果和实际配置；如果使用多个 Fold 汇总，则按 Mean ± Std 报告，并按需要生成柱状图。
- 本文件只定义实验路线，不改变 `PROTOCOL.md` 中的数据划分、训练超参数、Fold 评估使用规则和隐私规则。
- 现有 `Step 01_I`–`Step 10_I` 为患者隔离路线，统一使用 `Patient Isolation = True`。
- 不隔离患者路线使用 `Step 01_N`–`Step 10_N`，统一使用 `Patient Isolation = False`。
- 每次执行必须明确指定 Step 和路线，例如 `Step 05_I` 或 `Step 05_N`；未指定路线时不得开始训练或评估。

## Step 01_I：确定图像分辨率

### 目标

确认后续所有正式实验统一使用的图像输入分辨率为 `384 × 384`。

### 要求

- 使用二分类任务。
- 使用 `M` 白光图像作为输入；本计划中的 `M` 即 White Baseline。
- 项目当前固定使用 `384 × 384`；本 Step 不重新开展其他分辨率训练。
- 其他模型和训练条件保持一致。
- 对固定的 `384 × 384` 配置，使用 `Fold Seed = 42` 生成固定五折划分，并分别使用 `Training Seeds = [42, 3407, 2026]` 完整运行五个 Fold。
- 记录已确定 `384 × 384` 的依据，并在实验配置中固定该分辨率。

### 输出

- `384 × 384` 的固定分辨率记录。
- 后续实验全部固定使用该分辨率。
- Accuracy、Macro-F1 和适用的辅助指标柱状图。

## Step 02_I：二分类单通道消融实验

### 目标

判断不同图像通道单独使用时，对轻重程度二分类的有效性，找出具有较强分类信息的通道。

### 要求

- 使用 Step 1 确定的分辨率。
- 分别使用以下五个通道进行单独训练：`M`、`MB`、`MP`、`MR`、`MUV`。
- 本 Step 的通道集合固定为上述五个通道；如数据审计发现通道缺失，必须按 `PROTOCOL.md` 在当前 Fold 的训练集或评估集内过滤并报告，不得用其他未指定通道替代。
- 所有通道使用完全相同的训练策略。
- `M`（White）作为后续多通道实验的基础对照。

### 输出

- 各单通道性能对比。
- 柱状 Benchmark 图。
- 分析哪些通道优于或接近 White。

## Step 03_I：二分类多通道实验

### 目标

判断多通道输入是否比单独使用白光具有实际增益。

### 要求

- 使用 Step 2 的 `M`（White）结果作为 Baseline。
- 仅从 `M`、`MB`、`MP`、`MR`、`MUV` 中构造并预先列明多通道组合，包括需要验证的全部通道输入。
- 除输入通道组合外，其余条件保持一致。
- 重点比较多通道相对白光的性能变化。
- 不得因为通道缺失而重新划分五 Fold；只能在当前 Fold 内过滤。

### 输出

- White 与不同多通道方案的 Benchmark 对比。
- 判断多通道是否值得继续使用。
- 找到表现较好的多通道方案，并记录选择依据。

## Step 04_I：二分类 DeepLIFT + 通道扰动

### 目标

解释多通道模型依赖哪些通道，并判断每个通道对最终预测的实际贡献。

### 要求

- 基于 Step 3 中预先规定或经用户确认的代表性多通道模型进行分析。
- 使用 DeepLIFT 分析不同通道和图像区域的重要性。
- 对不同通道分别进行扰动或消融，观察模型性能变化。
- 扰动方法在同一组实验中保持一致。
- 通道扰动不得改变原始五 Fold 划分。
- 代表性模型的选择不得使用未预先规定的评估 Fold 表现规则；如需根据评估结果选择，必须先报告并获得明确的协议偏离批准。

### 输出

- DeepLIFT 可视化结果。
- 各通道扰动后的性能变化。
- 不同通道重要性的结论。

## Step 05_I：五分类单通道实验

### 目标

将任务切换为原始 0–4 五分类，判断不同通道对于精细严重程度分级的能力。

### 要求

- 使用已经确定的图像分辨率。
- 使用 Hard Label。
- 重复单通道对比实验。
- 重点关注相邻等级之间的误判情况。

### 输出

- 五分类单通道 Benchmark。
- White 作为后续五分类多通道实验的 Baseline。
- 分析不同通道对不同严重程度等级的区分能力。
- Accuracy、Macro-F1、MAE、Per-class 指标和混淆矩阵相关报告。

## Step 06_I：五分类多通道实验

### 目标

判断多通道是否能够改善五分类表现。

### 要求

- 使用 Step 5 的 White 结果作为 Baseline。
- 测试预先列明的合理多通道组合。
- 其他实验条件保持一致。
- 重点观察 Macro-F1、Accuracy 和等级误差的变化。
- 重点检查相邻类别之间的混淆变化。

### 输出

- 五分类 White 与多通道 Benchmark。
- 找到表现较好的多通道方案，并记录选择依据。
- 分析多通道是否减少相邻类别误判。

## Step 07_I：五分类 DeepLIFT + 通道扰动

### 目标

解释五分类模型在不同严重程度判断中依赖哪些通道。

### 要求

- 基于 Step 6 中预先规定或经用户确认的代表性多通道模型。
- 使用 DeepLIFT 分析不同类别下的通道贡献。
- 使用通道扰动验证各通道的重要性。
- 扰动方法、评价指标和数据划分保持一致。
- 代表性模型的选择不得使用未预先规定的评估 Fold 表现规则；如需根据评估结果选择，必须先报告并获得明确的协议偏离批准。

### 输出

- 五分类 DeepLIFT 结果。
- 通道扰动 Benchmark。
- 分析不同严重程度是否依赖不同通道。

## Step 08_I：五分类 Soft Label 单通道实验

### 目标

判断 Soft Label 是否能够改善五分类中相邻等级边界模糊的问题。

### 要求

- Soft Label 只用于五分类。
- 其他条件与 Step 5 保持一致。
- 重复单通道实验。
- 与 Hard Label 单通道结果直接比较。
- Label Strategy 是本 Step 相对于对应 Hard Label 实验的主要变量。

### 输出

- Hard Label vs Soft Label 对比。
- 判断 Soft Label 对不同通道是否有稳定提升。

## Step 09_I：五分类 Soft Label 多通道实验

### 目标

判断 Soft Label 与多通道结合后是否能进一步提高分类效果。

### 要求

- 使用 Soft Label。
- 重复多通道实验。
- 同时和 Hard Label 对应实验进行比较。
- 除 Label Strategy 和预先指定的输入通道组合外，其余条件保持一致。

### 输出

- Soft Label 条件下的多通道 Benchmark。
- Hard Label vs Soft Label 对比。
- 找到最终较优的输入和标签方案，并记录选择依据。

## Step 10_I：五分类 Soft Label DeepLIFT + 通道扰动

### 目标

判断使用 Soft Label 后，模型对不同通道的依赖是否发生变化。

### 要求

- 选择预先规定或经用户确认的代表性 Soft Label 多通道模型。
- 重复 DeepLIFT 和通道扰动分析。
- 与 Hard Label 的解释性结果进行对比。
- 代表性模型的选择不得使用未预先规定的评估 Fold 表现规则；如需根据评估结果选择，必须先报告并获得明确的协议偏离批准。

### 输出

- Soft Label DeepLIFT 结果。
- Soft Label 通道扰动结果。
- 总结 Hard Label 与 Soft Label 下模型关注通道的差异。

## MR ROI 实验扩展路线

本路线研究 MR 图像中人工标注的 `red` ROI 对 CEA 五分类性能的影响。所有实验开始前必须完整读取并遵守 `PROTOCOL.md`。

ROI 路线以已有 MR 单通道五分类实验为严格 Baseline。除当前 Step 明确改变的 ROI 输入方式或 Label Strategy 外，其余样本集合、Fold 划分、图像预处理、Normalize、数据增强、模型、训练超参数和评价方式均与对应 MR 单通道 Benchmark 保持一致。

### ROI 坐标换算与数据准备规则

LabelMe JSON 中的 polygon 坐标基于原始 MR 图像 `3448 × 4600`。当前 MR Benchmark 的固定预处理为：

```text
3448 × 4600
→ 等比例 Resize 为 288 × 384
→ 左右各 Padding 48 px
→ 384 × 384
```

所有 polygon 顶点统一按照以下规则转换：

```text
x_new = x × 288 / 3448 + 48
y_new = y × 384 / 4600
```

转换后的坐标裁剪到 `384 × 384` 图像边界后栅格化。一个样本内所有标签严格为 `red` 的 polygon 取并集，生成二值 Mask：ROI = 1，非 ROI = 0。坐标转换只在 ROI 数据准备阶段执行一次，后续实验直接使用转换后的 ROI 数据。

ROI 样本资格和配对 Baseline 规则：

- 通过 JSON 内部 `imagePath` 与 MR 图像建立映射，不依赖文件排列顺序；
- JSON 缺失、图像映射失败或没有有效 `red` polygon 的样本不进入 ROI 实验；
- ROI 实验沿用既有固定 Fold 名单，不因 ROI 资格过滤重新划分 Fold；
- 所有 ROI 输入、对应 Whole Image Baseline 和 Hard/Soft Label 对比必须使用完全相同的 ROI 有效样本集合；
- 记录每个 Fold 的有效训练样本数和评估样本数，以及被排除样本的原因；
- ROI Mask、Bounding Box、20% Context、ROI 面积比例和每个 Fold 的 Training Mean Fill 均保存为可追溯的匿名化元数据；
- ROI 可视化只用于本地 QA，不写入 Git，不在最终 HTML 报告中嵌入原始医学图像或患者级敏感数据。

Mean Fill 规则：

- 每个 Fold 仅使用该 Fold 的完整训练集 MR 图像计算 Mean Fill；
- Evaluation Fold 不参与 Mean Fill 计算；
- Mean Fill 在现有 `384 × 384` MR 输入、ImageNet Normalize 之前计算，并沿用当前 MR 的通道转换和 Normalize；
- 训练和评估时，ROI Mask 必须与 MR 图像执行完全一致的几何数据增强；
- ROI 数据准备和所有输入构造不得改变原始标签、Fold Seed、Training Seed 或患者隔离规则。

### Step 11_I：MR ROI 数据准备

#### 目标

建立与 `384 × 384` MR 图像对应的 ROI 数据，为后续 Mask、Bounding Box、Context、Background Only 实验准备输入。

#### 要求

- 完成 JSON-MR 映射检查和 ROI 坐标转换检查；
- 生成二值 ROI Mask 和所有 `red` polygon 的最小外接 Bounding Box；
- 生成每个 Fold 的 Training Mean Fill；
- 统计 ROI 数量、面积、面积比例、Bounding Box 尺寸及有效/排除样本数量；
- 生成本地 ROI 可视化 QA；
- 本 Step 不进行模型训练。

#### 输出

- ROI manifest、映射检查和数据完整性报告；
- ROI Mask、Bounding Box、面积统计和 Fold Mean Fill；
- 排除样本及排除原因；
- 本地可视化 QA 结果。

### Step 12_I：MR ROI Mask Hard Label 实验

#### 目标

判断仅保留人工标注 ROI 图像信息后，CEA 五分类性能如何变化。

#### 要求

- 使用 Hard Label；
- ROI 内保留原始 MR 像素，ROI 外使用当前 Fold Training Mean Fill；
- 输入定义为：

```text
ROI Mask Input = MR × Mask + Training Mean × (1 - Mask)
```

- 使用相同 ROI 有效样本集合上的 MR Hard Label Whole Image 结果作为 Baseline；
- 除 ROI 输入方式外，其余条件保持不变。

#### 输出

- MR Whole Image 与 MR ROI Mask 对比；
- Accuracy、Macro-F1、MAE、Per-class 指标、混淆矩阵和 Mean ± Std；
- 按 `PROTOCOL.md` 生成当前 Step 的 CSV、JSON、柱状图和 HTML 报告。

### Step 13_I：MR ROI Crop Hard Label 实验

#### 目标

判断仅使用病灶所在局部区域时，CEA 五分类性能如何变化。

#### 要求

- 使用 Hard Label；
- 根据所有 `red` ROI 的最小外接 Bounding Box 从 `384 × 384` MR 图像中裁剪；
- Crop 后使用现有无形变 `ResizePad` 恢复到 `384 × 384`；
- Crop 内保留真实病灶及 Bounding Box 内正常皮肤，不使用 Mask 或 Mean Fill；
- 使用相同 ROI 有效样本集合上的 MR Hard Label Whole Image 结果作为 Baseline；
- 除 ROI Crop 输入方式外，其余条件保持不变。

#### 输出

- MR Whole Image 与 MR ROI Crop 对比；
- Accuracy、Macro-F1、MAE、Per-class 指标、混淆矩阵和 Mean ± Std；
- 按 `PROTOCOL.md` 生成当前 Step 的 CSV、JSON、柱状图和 HTML 报告。

### Step 14_I：MR ROI + 20% Context Hard Label 实验

#### 目标

判断病灶周围正常皮肤是否能够提供额外的 CEA 分级信息。

#### 要求

- 使用 Hard Label；
- 在 Step 13 的 Bounding Box 基础上，上、下、左、右分别扩张原 Width / Height 的 `20%`；
- 超出 `384 × 384` 图像边界时截断；
- Context 比例固定为 `20%`，裁剪后使用现有无形变 `ResizePad` 恢复到 `384 × 384`；
- 使用真实 MR 图像，不使用 Mask 或 Mean Fill；
- 其他条件保持不变。

#### 输出

- MR Whole Image、MR ROI Crop、MR ROI + 20% Context 对比；
- Accuracy、Macro-F1、MAE、Per-class 指标、混淆矩阵和 Mean ± Std；
- 按 `PROTOCOL.md` 生成当前 Step 的 CSV、JSON、柱状图和 HTML 报告。

### Step 15_I：MR Background Only Hard Label 实验

#### 目标

判断去除人工标注 ROI 后，MR 剩余区域是否仍具有 CEA 分类信息。

#### 要求

- 使用 Hard Label；
- ROI 区域使用当前 Fold Training Mean Fill，非 ROI 区域保留原始 MR 图像；
- 输入定义为：

```text
Background Only = MR × (1 - Mask) + Training Mean × Mask
```

- 使用相同 ROI 有效样本集合上的 MR Hard Label Whole Image 结果作为 Baseline；
- 除输入方式外，其余条件保持不变。

#### 输出

- MR Whole Image 与 MR Background Only 对比；
- Accuracy、Macro-F1、MAE、Per-class 指标、混淆矩阵和 Mean ± Std；
- 按 `PROTOCOL.md` 生成当前 Step 的 CSV、JSON、柱状图和 HTML 报告。

## MR ROI Soft Label 路线

Soft Label 固定为：

```text
Class 0 → [0.9, 0.1, 0,   0,   0]
Class 1 → [0.1, 0.8, 0.1, 0,   0]
Class 2 → [0,   0.1, 0.8, 0.1, 0]
Class 3 → [0,   0,   0.1, 0.8, 0.1]
Class 4 → [0,   0,   0,   0.1, 0.9]
```

除 Label Strategy 外，Soft Label 实验必须与对应 Hard Label ROI 实验保持一致，包括 ROI 有效样本集合、Fold、输入构造、Mean Fill、训练条件和评价方式。

### Step 16_I：MR ROI Mask Soft Label 实验

- 使用固定 Soft Label 规则；
- ROI Mask 输入方式与 Step 12_I 完全一致；
- 与相同 ROI 有效样本集合上的 Hard Label ROI Mask 结果比较；
- 输出 Hard Label vs Soft Label 的 Accuracy、Macro-F1、MAE、Per-class 指标、混淆矩阵、Mean ± Std、柱状图和 HTML 报告。

### Step 17_I：MR ROI Crop Soft Label 实验

- 使用固定 Soft Label 规则；
- ROI Crop 输入方式与 Step 13_I 完全一致；
- 除 Label Strategy 外，其余条件与 Step 13_I 保持一致；
- 输出 Hard Label vs Soft Label ROI Crop 的完整指标和 HTML 报告。

### Step 18_I：MR ROI + 20% Context Soft Label 实验

- 使用固定 Soft Label 规则；
- Context 输入方式与 Step 14_I 完全一致，四边各扩张 `20%`；
- 除 Label Strategy 外，其余条件与 Step 14_I 保持一致；
- 输出 Hard Label vs Soft Label ROI + Context 的完整指标和 HTML 报告。

### Step 19_I：MR Background Only Soft Label 实验

- 使用固定 Soft Label 规则；
- Background Only 输入方式与 Step 15_I 完全一致；
- 除 Label Strategy 外，其余条件与 Step 15_I 保持一致；
- 输出 Hard Label vs Soft Label Background Only 的完整指标和 HTML 报告。

### ROI 路线固定执行与输出规则

- Step 11_I 仅做 ROI 数据准备；Step 12_I–Step 19_I 每个 Step 使用 3 个 Training Seed × 5 个 Fold，共 15 次训练与评估；
- 输出目录分别为 `isolated\\step_11_I` 至 `isolated\\step_19_I`；
- 每个正式训练 Step 必须包含配置、日志、Fold-level 指标、`last.pth`、原始 CSV / JSON、Mean ± Std、Accuracy / Macro-F1 / MAE 图表、Per-class 图表、混淆矩阵和自包含 HTML 报告；
- 不得使用评估 Fold 表现选择 Epoch、Checkpoint、Fold、Training Seed 或代表性模型；
- 结果目录和 Git 中不得保存原始医学图像、患者身份信息、全量缓存或未经批准的 ROI 敏感数据。

## 不隔离患者平行训练路线（Step 01_N–Step 10_N）

本路线与对应的患者隔离路线使用相同的任务目标、输入通道、分辨率、模型、训练超参数、五 Fold 评估方式和输出要求。唯一的数据划分差异是：固定使用 `Patient Isolation = False`，按样本进行五折划分；同一患者可以出现在不同 Fold，不要求患者 ID 在训练 Fold 与评估 Fold 之间完全隔离。每次运行使用 `Fold Seed = 42` 生成固定划分，并分别使用 `Training Seeds = [42, 3407, 2026]`、`Batch Size = 32`，完整执行五个 Fold。

### Step 01_N：确定图像分辨率

- 对应隔离路线 Step 01_I；目标、训练条件和输出要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 02_N：二分类单通道消融实验

- 对应隔离路线 Step 02_I；使用相同的五个单通道和输出要求。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 03_N：二分类多通道实验

- 对应隔离路线 Step 03_I；多通道组合、Baseline 和比较要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 04_N：二分类 DeepLIFT + 通道扰动

- 对应隔离路线 Step 04_I；代表性模型、扰动方法和输出要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 05_N：五分类单通道实验

- 对应隔离路线 Step 05_I；使用 Hard Label、单通道对比和五分类指标要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 06_N：五分类多通道实验

- 对应隔离路线 Step 06_I；Baseline、多通道组合和相邻类别混淆分析要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 07_N：五分类 DeepLIFT + 通道扰动

- 对应隔离路线 Step 07_I；解释性分析、扰动方法和输出要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 08_N：五分类 Soft Label 单通道实验

- 对应隔离路线 Step 08_I；Soft Label、单通道实验和 Hard Label 对比要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 09_N：五分类 Soft Label 多通道实验

- 对应隔离路线 Step 09_I；Soft Label、多通道组合和比较要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

### Step 10_N：五分类 Soft Label DeepLIFT + 通道扰动

- 对应隔离路线 Step 10_I；DeepLIFT、通道扰动和 Hard Label 对比要求完全相同。
- 使用 `Patient Isolation = False` 的样本级五折划分。

## MR ROI 不隔离患者平行路线（Step 11_N–Step 19_N）

本路线与 `Step 11_I–Step 19_I` 使用相同的 ROI 坐标换算、ROI 有效样本规则、输入构造、Mean Fill、Context 比例、Soft Label 规则、模型、训练超参数、评价指标和输出要求。

唯一的数据划分差异为：

- `Patient Isolation = False`；
- 沿用不隔离患者路线已有的固定样本级五折名单，不因 ROI 资格过滤重新划分 Fold；
- 对应的 MR Whole Image Baseline、ROI 实验和 Hard/Soft Label 对比使用相同的 ROI 有效样本集合；
- 输出目录为 `non_isolated\\step_11_N` 至 `non_isolated\\step_19_N`。

### Step 11_N：MR ROI 数据准备

对应 `Step 11_I`，仅完成 JSON-MR 映射、坐标转换、ROI Mask、Bounding Box、面积统计、排除原因和每个 Fold 的 Training Mean Fill，不进行模型训练。

### Step 12_N：MR ROI Mask Hard Label 实验

对应 `Step 12_I`，使用 ROI Mask 输入和 Hard Label，比较同一 ROI 有效样本集合上的 MR Whole Image Baseline。

### Step 13_N：MR ROI Crop Hard Label 实验

对应 `Step 13_I`，使用 ROI Bounding Box Crop，并通过现有无形变 `ResizePad` 恢复到 `384 × 384`。

### Step 14_N：MR ROI + 20% Context Hard Label 实验

对应 `Step 14_I`，使用四边各扩张 `20%` 的 Context Crop，并通过现有无形变 `ResizePad` 恢复到 `384 × 384`。

### Step 15_N：MR Background Only Hard Label 实验

对应 `Step 15_I`，ROI 区域使用当前 Fold Training Mean Fill，非 ROI 区域保留原始 MR 图像。

### Step 16_N：MR ROI Mask Soft Label 实验

对应 `Step 16_I`，仅将 Label Strategy 改为固定 Soft Label，其他条件与 `Step 12_N` 保持一致。

### Step 17_N：MR ROI Crop Soft Label 实验

对应 `Step 17_I`，仅将 Label Strategy 改为固定 Soft Label，其他条件与 `Step 13_N` 保持一致。

### Step 18_N：MR ROI + 20% Context Soft Label 实验

对应 `Step 18_I`，仅将 Label Strategy 改为固定 Soft Label，其他条件与 `Step 14_N` 保持一致。

### Step 19_N：MR Background Only Soft Label 实验

对应 `Step 19_I`，仅将 Label Strategy 改为固定 Soft Label，其他条件与 `Step 15_N` 保持一致。

每个 `Step 12_N–Step 19_N` 使用 3 个 Training Seed × 5 个 Fold，共 15 次训练与评估；完整保存配置、日志、Fold-level 指标、原始 CSV / JSON、Mean ± Std、柱状图、混淆矩阵和自包含 HTML 报告。Step 11_N 仅保存 ROI 数据准备和完整性检查结果。

## 最终目标

完成训练计划后，需要回答以下问题：

1. 什么图像分辨率最适合当前任务？
2. 哪些单通道最有分类价值？
3. 多通道相比 White 是否真正提升性能？
4. 不同通道分别贡献了什么？
5. 二分类与五分类的困难差异在哪里？
6. Soft Label 是否能够改善五分类？
7. 最终推荐哪一种输入方式和标签策略？
