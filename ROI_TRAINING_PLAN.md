# ROI 医学图像分类训练计划

## 0. 与现有文档的关系

本文件是 `PROTOCOL.md` 与 `TRAINING_PLAN.md` 之外的第三层实验路线，只定义 ROI 相关 Benchmark 的目标、变量和输出要求。执行优先级与既有文档层级一致：

1. `PROTOCOL.md`：实验硬规则，仍然完整适用，开始前必须完整读取。
2. `TRAINING_PLAN.md`：已完成的 Step 01–10 主线实验，其结果作为本计划的 Baseline 锚点。
3. 本文件（`ROI_TRAINING_PLAN.md`）：ROI 实验路线。
4. 当前 Prompt：本次任务范围，必须明确指定执行哪个 Step（例如 `Step R02_I`）。

### 0.1 协议增补项（需用户批准后生效）

本计划引入以下经预先声明的协议变化，其余一切硬规则（数据范围、排除规则、五折划分、Fold Seed = 42、Training Seeds = [42, 3407, 2026]、ResNet50、AdamW、Learning Rate = 1e-4、Weight Decay = 1e-4、Batch Size = 32、Epoch = 50、无 Scheduler、无 Early Stopping、CrossEntropyLoss、评价指标、Mean ± Std 汇总、图表与 HTML 报告规则）保持不变：

1. 新增 Benchmark 变量：`Input Region`（ROI 策略）。ROI Benchmark 的唯一主要变量为输入区域构造方式：`Full Image`（全图，Baseline，复用主线结果）、`ROI Crop`（ROI 联合外接框裁剪）、`ROI Mask`（ROI 外区域中性填充）。
2. 预处理链扩展为：`384 × 384 基础图 → ROI 区域处理（裁剪或掩码）→ 等比例 Resize → Padding → 384 × 384 正方形输入`。等比 Resize、禁止拉伸、Padding 一致性等规则不变；ROI 处理只改变"输入哪块区域"，不改变 Resize / Padding / Normalize / 增强策略本身。
3. 新增根目录共享模块 `roi_dataset.py`（继承 / 导入 `dataset.py`，提供 ROI 读取、裁剪、掩码与掩码同步增强），Step 文件不得复制其实现；`dataset.py` 与 `model.py` 保持不动。
4. 项目结构扩展两个结果目录：`roi_isolated\step_RXX_I\` 与 `roi_non_isolated\step_RXX_N\`，目录内部结构、文件命名与 `PROTOCOL.md` §16 完全一致（`step_R02_I.py`、`config_step_R02_I.json`、`fold_01`–`fold_05` 等）。
5. ROI 掩码类实验的数据增强同步规则：图像与 ROI 掩码必须使用同一次随机抽样的翻转、旋转参数（同一 RNG 决策，双份施加）；亮度 / 对比度 Color Jitter 只施加于图像。该规则是 `PROTOCOL.md` §5 "Train 数据增强策略保持固定"在掩码场景下的确定性实现。

以上增补项在首次执行任何 ROI Step 前必须获得用户明确批准，并记录在对应 `config_step_RXX_*.json` 的 `protocol_amendments` 字段中。

## 1. ROI 数据资产与坐标约定

### 1.1 数据资产

- ROI 标注为 labelme JSON 多边形，唯一标签 `red`（红斑 / 病灶区域），来源于 MR 通道原图的人工标注。
- 训练用副本：`roi_384_json\MRxxxx.json`，坐标已从原图（3448 × 4600）映射到 384 × 384 基础图坐标系，`imageData` 已剥离。坐标映射公式：`s = 384 / 4600`，`x' = x·s + (384 − 3448·s)/2`，`y' = y·s`；该映射已经 SIFT 内点仿射拟合（残差中位数 0.1 px）与对齐灰度相关性（r = 0.985）双重验证。原始 JSON 保留在数据目录，不进入 Git。
- 基础图像：`all_384` 中的 MR 通道 384 × 384 PNG，即主线实验使用的同一套图像。

### 1.2 覆盖与排除（2026-09-18 审计）

- 目录内 MR PNG 共 996 张，ROI JSON 共 998 个（多出者为 MR0069、MR0296，均无对应 PNG 且属于全局排除编号）。
- 769、770 不在目录中。执行全局排除 {69, 296, 769, 770} 后：可用 PNG 996 张、可用 JSON 996 个，两集合完全一致。
- 结论：**每一个进入实验的 MR 样本都有 ROI 标注**。ROI 实验与主线 MR 实验使用完全相同的有效样本集合与五折划分，不需要因 ROI 缺失做任何样本过滤，结果可直接横向比较。
- 标注统计：每图多边形 1–15 个（均值 5.62，共 5600 个）；ROI 面积占全图比例最小 0.05%、中位数 11.1%、最大 23.6%。

### 1.3 已知风险

- 极端微 ROI：最小面积样本（约 8 × 9 px）裁剪后放大到 384 存在约 40 倍上采样，纹理信息接近丢失。默认不做过滤（保持与 Baseline 样本一致），在 Step R01 中输出裁剪尺寸分布并在报告中明示该风险；如需过滤必须作为协议偏离单独报批。
- 384 坐标系下的多边形边界存在锯齿 / 混叠，掩码填充以像素中心点在多边形内为准（`matplotlib.path` 或等价实现），该判定方法在本计划所有 Step 中保持一致。
- 标注质量（边界贴合度）未经第二人复核，视为既定输入，不做修改。

## 2. 使用规则

- 所有实验开始前必须先完整读取并遵守 `PROTOCOL.md`，再读取本文件中对应的 Step。
- 本文件不是一次性锁定的执行顺序；每个 Step 完成后必须根据实际结果审核后续步骤，后续步骤可以调整、暂停或停止。
- 任何后续 Step 的目标、输入组合、主要变量或输出要求发生变化，都必须在开始该 Step 前由用户确认并记录。
- 当前 Prompt 必须明确本次执行的 Step 和路线（例如 `Step R03_I` 或 `Step R06_N`）；未指定时不得开始训练或评估。
- 除当前 Step 明确要求改变的变量（`Input Region` 或该 Step 声明的变量）外，其余训练设置保持不变。
- 如果本文件与 `PROTOCOL.md` 冲突，以 `PROTOCOL.md` 为准；必须先报告冲突，不得自行解释或覆盖协议。§0.1 列出的增补项是唯一预先声明的例外。
- 每个 Step 开始前必须完成 `PROTOCOL.md` §17 数据完整性检查外加本文件 §3 的 ROI 附加检查；检查失败立即停止。
- 每个正式 Benchmark Step 完成后保存原始 CSV / JSON、Mean ± Std 汇总和 `PROTOCOL.md` §13 规定的柱状图结果，并记录实际配置。
- 解释性分析 Step 必须保存原始分析结果和实际配置；如使用多个 Fold 汇总，按 Mean ± Std 报告。
- ROI 实验只使用 `MR` 单通道；`M`、`MB`、`MP`、`MUV` 不参与，除非用户明确批准扩展。
- Baseline 引用规则：全图 MR 的二分类 / 五分类结果分别原样引用 `step_02_I` / `step_05_I`（N 路线引用 `step_02_N` / `step_05_N`）的 15 条 Fold-level 记录，不重新训练；引用必须在 config 和报告中注明来源与批准记录。

## 3. ROI 附加数据完整性检查

每次 ROI 实验开始前，除 `PROTOCOL.md` §17 外必须额外检查：

- 每个进入当前 Fold 的样本都存在 `roi_384_json\MR<sample_id>.json` 且至少含 1 个 `red` 多边形。
- 多边形坐标全部位于 [0, 384] 范围内。
- `roi_dataset.py` 生成的裁剪 / 掩码尺寸合法（裁剪框非空、掩码非全零、非全一）。
- ROI 引入后样本集合与主线 MR 实验完全一致（不因 ROI 增删任何样本）。
- 五折划分文件与主线实验逐样本一致（同一 Fold Seed = 42 产物，只读复用，禁止重新划分）。
- 每个 Fold 记录 ROI 多边形数分布与面积分布，确认各 Fold 之间无系统性偏差。

## 4. 患者隔离路线（Step R01_I–R08_I）

### Step R01_I：ROI 数据审计与基线锚定

#### 目标

在不训练任何模型的前提下，建立 ROI 实验的数据基线：确认 ROI 资产完整、统计标注分布、锁定与主线实验的可比性，并向用户报告所有需要在后续 Step 前决策的事项。

#### 要求

- 只做数据审计与统计，不运行训练。
- 验证 §1.2 的覆盖结论（排除后 996 PNG ↔ 996 JSON 完全一致）。
- 统计并保存：每图多边形数分布、ROI 面积占比分布、ROI 联合外接框尺寸分布（含按 label 0–4 分层的版本）、裁剪后相对上采样倍数分布。
- 验证主线五折划分文件可直接复用：逐样本比对 Fold 归属，报告一致率（要求 100%）。
- 生成 `roi_dataset.py` 的冒烟测试：随机抽取若干样本渲染裁剪 / 掩码输出，人工核验边界正确性；渲染图只保存在本地非 Git 目录（隐私规则），核验结论写入审计报告。
- 锚定 Baseline：从 `step_02_I`、`step_05_I` 导入 MR 全图 15 条 Fold-level 记录，保存为本路线 Baseline 档案文件。

#### 输出

- `roi_isolated\step_R01_I\` 下的审计 CSV / JSON、分布图（匿名化统计图）、Baseline 档案与审计报告。
- 向用户报告的决策点清单：极端微 ROI 是否保留（默认保留）、裁剪外扩边距是否为 0（默认 0）、掩码填充值取灰色 128（默认）等，等待确认后进入 R02。

### Step R02_I：二分类 ROI 裁剪实验

#### 目标

判断模型输入只保留 ROI 联合外接框区域时，对轻重程度二分类的影响：病灶区域聚焦是否优于全图输入。

#### 要求

- 使用二分类任务（0、1、2 → Class 0；3、4 → Class 1）。
- 输入构造：384 基础图上取全部 `red` 多边形的联合外接框（`margin = 0`，整数化后 clamp 到图像边界），裁剪后经统一的等比 Resize + Padding 到 384 × 384。
- 唯一允许改变的变量：`Input Region`（Full Image → ROI Crop）。Full Image 结果引用 Step R01_I 锚定的 `step_02_I` MR 记录，不重新训练。
- 数据增强在裁剪之后施加，策略与主线完全一致（水平翻转 0.5、旋转 ±7°、亮度 / 对比度 ±0.10）。
- 使用 `Patient Isolation = True` 的主线五折划分、Fold Seed = 42、Training Seeds = [42, 3407, 2026]，共 15 次运行。
- 附带记录（不用于选择，只用于报告）：按裁剪尺寸分层的性能分组统计、按 ROI 面积分层的分组统计。

#### 输出

- Full Image vs ROI Crop 的 Accuracy / Macro-F1 / Sensitivity / Specificity / AUC 横向柱状图（Mean ± Std，三 Training Seed 汇总）。
- 混淆矩阵、Per-class 指标。
- 裁剪尺寸与上采样倍数分布附图。
- 结论：ROI 裁剪是否带来增益、增益是否稳定。

### Step R03_I：二分类 ROI 掩码实验

#### 目标

在保留全图构图的前提下，将 ROI 之外的区域替换为中性填充，判断"去除背景干扰但保留空间上下文"是否优于全图与裁剪。

#### 要求

- 使用二分类任务。
- 输入构造：384 基础图上以像素中心点法将所有 `red` 多边形外的像素替换为固定中性灰 (128, 128, 128)；ROI 内像素保持原值。填充值在同一 Benchmark 内固定。
- 图像与 ROI 掩码必须使用同一次随机决策的翻转与旋转（同一参数双份施加）；亮度 / 对比度只施加于图像。评估 Fold 禁止随机增强。
- 唯一允许改变的变量：`Input Region`（Full Image → ROI Mask）。Baseline 同样引用 R01_I 锚定记录。
- 五折划分、Seeds、超参数与 R02_I 完全一致。

#### 输出

- Full Image vs ROI Crop vs ROI Mask 三方案对比柱状图（Mean ± Std）。
- 混淆矩阵与 Per-class 指标。
- 结论：三种输入区域构造的排序及是否值得在五分类中继续。

### Step R04_I：二分类 ROI 量化特征基线（辅助性 Feature Benchmark）

#### 目标

用 ROI 直接计算的手工量化特征 + 简单分类器，估计标注本身携带的判别信息量，为 CNN 结果提供参照下界。

#### 要求

- 本 Step 是预先声明的 Model 变量 Benchmark（协议增补项之一）：分类器为 Logistic Regression（scikit-learn，L2，默认超参数），不使用 ResNet50；其余数据规则不变。
- 特征从 ROI 与 MR 384 图计算，预先固定为：ROI 面积占比、多边形数量、ROI 内灰度 / RGB 各通道均值与标准差、ROI 内最大连通区域占比。特征定义在 Step 开始前锁定，禁止看到结果后增删。
- 使用与主线完全相同的五折划分与三个 Training Seed（逻辑回归确定性收敛，三次运行结果应一致；如实记录一致性检查）。
- 五分类与二分类标签规则动态生成，不修改原始标签。
- 结果单独成图，只作为参照线呈现，不与 CNN Benchmark 混合排序。

#### 输出

- 特征定义清单与特征值分布统计。
- 15 条 Fold-level 记录与 Mean ± Std 汇总。
- 参照结论：标注信息量相对于 CNN 全图 / ROI 结果的位置。

### Step R05_I：二分类 ROI 解释性与区域扰动

#### 目标

量化模型对 ROI 区域的实际依赖：预测是否真的由标注的病灶区域驱动。

#### 要求

- 代表性模型按预先规定规则确定：Training Seed = 42、fold_01 的 Checkpoint；在 R02_I 与 R03_I 完成后，取两者中 Macro-F1 最终 Mean 较高者的输入方案（该选择规则预先固定，只用于解释性分析，不构成模型选择）。另取主线 `step_02_I` MR 全图模型（Seed 42、fold_01）作对照，只读复用其 Checkpoint。
- 区域扰动（对全图模型）：ROI 内填充中性灰（erase-ROI）与 ROI 外填充中性灰（keep-ROI-only），分别评估 Macro-F1 变化；扰动在评估 Fold 上全样本执行。
- DeepLIFT：对代表性模型计算归因图，报告归因质量落在 ROI 内的比例（逐样本计算后按 Fold 汇总 Mean ± Std）。归因可视化只保存在本地非 Git 目录，报告中仅存匿名化数值统计。
- 扰动方法与评价指标在同一组实验内保持一致；不得改变五折划分。

#### 输出

- 扰动前后 Accuracy / Macro-F1 变化柱状图。
- DeepLIFT 归因-ROI 重叠比例汇总。
- 结论：模型依赖 ROI 的程度，与 R02/R03 的性能结论是否互相印证。

### Step R06_I：五分类 ROI 裁剪实验

#### 目标

将任务切换为原始 0–4 五分类，检验 ROI 聚焦对精细严重程度分级的作用。

#### 要求

- 使用 Hard Label 五分类，num_classes = 5。
- 输入构造、增强、划分、Seeds、超参数与 R02_I 完全一致。
- 唯一允许改变的变量：`Input Region`（Full Image → ROI Crop）。Baseline 引用 `step_05_I` MR 记录。
- 重点关注 MAE、相邻等级（0↔1、1↔2、2↔3、3↔4）混淆变化、Per-class Recall。

#### 输出

- Full Image vs ROI Crop 的 Accuracy / Macro-F1 / MAE 对比与混淆矩阵。
- 相邻等级误判变化分析。
- 结论：ROI 裁剪对五分类是否有增益，增益集中在哪些等级。

### Step R07_I：五分类 ROI 掩码实验

#### 目标

检验 ROI 掩码在五分类下的表现，并与 R06_I、全图 Baseline 三方比较。

#### 要求

- 与 R03_I 相同的输入构造与掩码同步增强规则；与 R06_I 相同的任务与评估要求。
- 唯一允许改变的变量：`Input Region`（Full Image → ROI Mask）。
- Baseline 引用 `step_05_I` MR 记录；与 R06_I 结果并列比较。

#### 输出

- 五分类三方案（Full Image / ROI Crop / ROI Mask）Benchmark 图。
- Per-class 指标、MAE、相邻等级混淆对比。
- 结论：确定五分类下较优的输入区域构造。

### Step R08_I：五分类 ROI 解释性与区域扰动

#### 目标

解释五分类模型在不同严重程度判断中对 ROI 的依赖，并与二分类解释性结果对照。

#### 要求

- 代表性模型规则与 R05_I 相同，对象换成 R06_I / R07_I 与 `step_05_I` MR 全图模型（Seed 42、fold_01）。
- 扰动与 DeepLIFT 流程与 R05_I 完全一致。
- 额外按预测类别分层报告归因-ROI 重叠比例，分析不同等级是否依赖不同区域。

#### 输出

- 五分类扰动 Benchmark 与 DeepLIFT 汇总。
- 二分类 vs 五分类解释性对照结论。

## 5. 不隔离患者平行训练路线（Step R01_N–R08_N）

本路线与对应的隔离路线使用相同的目标、输入构造、变量定义和输出要求，唯一差异是固定使用 `Patient Isolation = False` 的样本级五折划分（同样由 Fold Seed = 42 生成、与主线 N 路线逐样本一致）。Baseline 分别引用 `step_02_N` / `step_05_N` 的 MR 全图记录。每个 Step 的名称、目录与文件编号按 `roi_non_isolated\step_RXX_N` 命名。

- Step R01_N：ROI 数据审计与基线锚定（N 路线划分比对与 Baseline 导入）。
- Step R02_N：二分类 ROI 裁剪实验。
- Step R03_N：二分类 ROI 掩码实验。
- Step R04_N：二分类 ROI 量化特征基线。
- Step R05_N：二分类 ROI 解释性与区域扰动。
- Step R06_N：五分类 ROI 裁剪实验。
- Step R07_N：五分类 ROI 掩码实验。
- Step R08_N：五分类 ROI 解释性与区域扰动。

## 6. 候选扩展（未排入主线，启动前需用户单独确认）

- 多实例学习：每个 `red` 多边形独立裁剪为实例，MIL 池化后分类，可能缓解联合外接框的分辨率稀释问题。
- Soft Label × ROI：将主线 Step 08–09 的 Soft Label 策略与本计划较优输入方案结合。
- ROI 引导注意力：将多边形栅格化为软掩码作为注意力先验输入模型，属于 Model 变量 Benchmark，需另行协议增补。

## 7. 最终目标

完成本训练计划后，需要回答以下问题：

1. ROI 联合外接框裁剪相对全图输入，在二分类与五分类上是否提升 Macro-F1 / Accuracy / MAE？
2. ROI 掩码（保留上下文、去除背景）相对裁剪与全图的排序如何？
3. 增益（若有）更多来自病灶纹理细节，还是来自去除背景 / 非病灶区域干扰（裁剪与掩码结果对照）？
4. 模型归因（DeepLIFT）与区域扰动是否证实模型实际依赖 ROI 区域？
5. 人工标注的 ROI 量化特征本身携带多少判别信息（相对 CNN 的参照位置）？
6. 极端微 ROI 与大面积 ROI 样本上，上述结论是否稳定（分层附表）？
7. 最终推荐的输入构造：Full Image、ROI Crop 还是 ROI Mask，依据是什么？
