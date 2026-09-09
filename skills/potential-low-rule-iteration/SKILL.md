---
name: potential-low-rule-iteration
description: "Use when: analyzing a new user_profiles.jsonl batch with low-score label CSVs; assessing current potential-low-score rule applicability; discovering explainable candidate rules; iterating YAML rules, weights, and thresholds; or deciding whether a potential-low-score rule version can be released. 适用于新批次潜在低分规则的适用性分析、候选规则发现、调优迭代和发布判断。"
---

# 潜在低分规则迭代

当新批次已导出为与 `data/user_profiles.jsonl` 相同的结构时，使用本工作流。该流程维护可解释、由 YAML 驱动的规则系统，不训练或部署统计模型。

## 输入

分析前应具备以下输入：

- 新的 `user_profiles.jsonl` 格式文件，包含 `phone_id`、`basic_info`、`recent_metrics` 和 `user_journey`。
- 包含 `phone_id,is_low_score` 的优化标签 CSV，作为校准集。
- 格式相同、独立提供的可选评测标签 CSV。不得使用它选择规则、权重或阈值。
- 当前 `nps_analysis/protential_low_score/rules.yaml` 的副本，作为基线规则版本。
- 每次迭代使用单独的输出目录，例如 `data/potential-low-score/20260909_v1`。

不得把任一标签 CSV 中缺失的画像用户作为负样本。文件关联前应规范化并校验 `phone_id`。

## 目录与命名约定

所有新批次使用 `YYYYMMDD` 作为批次 ID。运行本 Skill 前，按以下目录结构准备和查找文件：

```text
data/
  incoming/
    <batch-id>/
      user_profiles.jsonl
      calibration_labels.csv
      evaluation_labels.csv
  rule-versions/
    rules_<batch-id>_baseline.yaml
    rules_<batch-id>_v1.yaml
    rules_<batch-id>_v2.yaml
  potential-low-score/
    <batch-id>_baseline/
    <batch-id>_v1/
    <batch-id>_v2/
```

文件查找及创建规则：

1. 从 `data/incoming/<batch-id>/user_profiles.jsonl` 读取本批用户画像。
2. 从同目录的 `calibration_labels.csv` 读取校准标签；该文件是必需输入。
3. 优先读取同目录的 `evaluation_labels.csv`。文件不存在时继续执行，但将结果明确标记为“仅校准集结果，不可发布”。
4. 首次运行前，将当前 `nps_analysis/protential_low_score/rules.yaml` 复制为 `data/rule-versions/rules_<batch-id>_baseline.yaml`。基线和后续迭代都使用此目录下的版本化规则，不直接修改项目默认规则文件。
5. 基线输出固定写入 `data/potential-low-score/<batch-id>_baseline/`；每次迭代的规则 YAML 与输出目录使用相同版本号，例如 `rules_<batch-id>_v1.yaml` 对应 `<batch-id>_v1/`。
6. 不覆盖任何已有版本目录或 YAML。若路径已存在，应使用下一个未占用版本号。
7. 上一批次存在时，将其输出目录作为分布、命中率和指标漂移的对比基线；不得把上一批标签混入当前批校准集或评测集。

标签文件必须使用 UTF-8 或 UTF-8 BOM 编码，首行至少包含 `phone_id,is_low_score`，其中 `is_low_score` 仅允许 `0` 或 `1`。校准集与评测集的 `phone_id` 应无交集；如有交集，须在报告中列出重叠数并要求数据方拆分后再做发布判断。

## 当前系统约束

- 规则配置在 `nps_analysis/protential_low_score/rules.yaml`。
- 每条规则对用户 $u$ 产生二元命中结果 $H_{u,i}\in\{0,1\}$。
- 已启用且命中的规则以优化后的权重 $w_i$ 参与内部累计：

$$
S_u=\sum_i w_iH_{u,i}
$$

- 仅当 $S_u\geq T$ 时，用户被预测为潜在低分。
- 每条规则的 `weight_candidates` 控制其可选贡献值；顶层 `threshold_candidates` 控制 $T$ 的候选值。
- `enabled: false` 会完全排除该规则。`enabled: true` 但优化后权重为 `0` 的规则仍可被度量，但不参与最终判定。
- 当前优化器为迭代式坐标搜索，不是穷举网格搜索。它只能在配置候选空间和起始点下找到可达的最优结果，不得描述为保证全局最优。
- 面向业务的预测 CSV 不得输出内部得分。

## 必须先运行基线

修改任何规则前，必须先使用当前基线规则运行新批次：

```powershell
.\.venv\Scripts\python.exe -m nps_analysis.protential_low_score.cli `
  --profiles "<new-profiles.jsonl>" `
  --labels "<calibration-labels.csv>" `
  --evaluation-labels "<evaluation-labels.csv>" `
  --output "<new-output-directory>" `
  --config "nps_analysis\protential_low_score\rules.yaml"
```

没有独立评测集时，省略 `--evaluation-labels` 并标记为“仅校准集结果”；该结果不可作为最终发布验证。

收集以下基线输出：

- `rule_optimization.json`：权重、阈值、校准指标、可选评测指标和规则影响。
- `calibration_prediction_detail.csv`：校准集用户及其 `TP/FP/FN/TN` 结果。
- `calibration_recalled_low_users.csv`：校准集真阳性用户。
- `evaluation_prediction_detail.csv`、`evaluation_recalled_low_users.csv`：提供评测集时的评测结果。
- `user_rule_hit_detail.csv`：每条命中规则、其选中权重及是否参与最终判定。

## 阶段一：适用性评估

调优前先评估新批次。产出简短的 `applicability_report.md` 或等价报告，包含：

1. 画像数、校准标签数、评测标签数、标签与画像交集数、低分率。
2. `basic_info`、月度指标、事件日期、`action`、`business`、`intent` 和新增事件字段的缺失率及类型检查。
3. 与上一批次比较事件动作、业务/意图分类、月度信号可用性及各启用规则的命中率分布。
4. 因引用字段或分类消失而无法评估的规则。
5. 命中率明显变化的规则，尤其是从接近零变为高覆盖，或反向变化的情况。

当规则输入字段缺失、命中人数过少无法评估，或命中率变化使历史效果不再可信时，应将其标为待复核。不得通过修改权重来掩盖数据映射变化。

## 阶段二：仅在校准集诊断现有规则

使用校准集生成候选改动，不得基于评测集选择改动。

### 分析假阳性

在 `calibration_prediction_detail.csv` 中筛选 `prediction_outcome=FP`，再关联 `user_rule_hit_detail.csv` 的命中规则。

优先检查符合下列任一条件的规则：

- 命中人数高但单规则 Precision 低。
- 选中权重大于零且 `without_precision_delta` 为负：移除规则后整体 Precision 更高。
- 单独即可跨过当前阈值的弱规则。
- 与强规则高度重叠但边际贡献很小。

每条候选优先尝试最小语义改动：收紧动作/分类/模式，缩小允许权重范围，在 `weight_candidates` 中加入 `0`；只有仍无帮助时才使用 `enabled: false`。

### 分析假阴性

在 `calibration_prediction_detail.csv` 中筛选 `prediction_outcome=FN`。将其画像与校准集 TN 用户比较，寻找 FN 常见、TN 少见且可解释的模式。

候选规则发现优先级如下：

1. 新的 `action + business + intent` 组合，使用 `journey_category_count` 表达。
2. 有界时间窗口内的重复事件，使用 `journey_count` 表达。
3. 有时序关系的事件对，使用 `journey_sequence` 表达。
4. 有明确边界的事件数值信号，使用 `journey_numeric_lt` 表达。
5. 月度费用/次数条件，使用 `monthly_positive_count` 或 `monthly_sum_gt` 表达。
6. 可解释的弱信号组合，使用 `all_of` 表达。

只使用画像中已结构化的字段。不得直接由 `raw_text` 推导生产规则；确有需要时，应先提升上游的 `business` 或 `intent` 分类。

## 候选规则准入

每条拟议规则写入 YAML 前都必须审核。在 `candidate_rules.md` 表格中记录以下证据：

- 规则 ID、风险类型、业务解释、DSL 条件和来源字段。
- 校准集命中数、TP 数、FP 数、单规则 Precision、单规则 Recall 和 Lift。
- 是否主要覆盖既有 FN 用户，或与已有规则重复。
- 对业务名单规模的预期影响。

Use these definitions:

$$
Precision(r)=\frac{TP_r}{TP_r+FP_r}
$$

$$
Lift(r)=\frac{Precision(r)}{\text{calibration low-score rate}}
$$

对命中数过少、缺乏清晰业务解释、Lift 弱、输入字段不稳定或几乎完全与已有规则重复的候选，应拒绝或暂缓。不得根据极小样本臆定数值阈值；应测试少量、可解释的候选集合。

## 阶段三：受控迭代

每次实验创建新的 YAML 版本和输出目录。每轮只改变一个维度：

1. 新增一条候选规则，或收紧/禁用一条现有规则。
2. 新规则或不确定规则保持 `enabled: true`，但在 `weight_candidates` 中包含 `0`，例如 `[0, 1, 2]`。
3. 使用与基线相同的模式和阈值候选，在校准集运行优化器。
4. 比较校准集 $TP$、$FP$、$FN$、$TN$、Precision、Recall、$F_1$、选中阈值、选中权重、名单量和逐规则影响。
5. 若校准结果改善了约定目标，则将冻结结果在评测集运行一次并记录指标。

同一轮不得同时改变多条规则、优化模式和阈值候选，否则无法归因指标改善来源。每次运行记录输入批次 ID、规则版本文件、输出目录、运行时间和与上一版本的唯一差异。

## 目标选择

迭代开始前确定优化模式：

| 业务需求 | YAML 模式 | 发布检查 |
| --- | --- | --- |
| 服务能力有限、假阳性代价高 | `precision` | 要求最低 Recall 和最大名单量 |
| 需广覆盖、漏掉真实低分代价高 | `recall` | 要求最低 Precision 和业务可承接名单量 |
| 一般均衡使用 | `balanced` | 在容量允许下优先更高的评测集 $F_1$ |

`balanced` 使用：

$$
F_\beta=(1+\beta^2)\frac{Precision\times Recall}{\beta^2\times Precision+Recall}
$$

低分标签不均衡，不得只优化 Accuracy。判定版本更优前，应先定义可接受的最低 Precision、最低 Recall 和最大预测用户数。

## 阶段四：评测与发布判断

评测数据仅用于观察。不得根据评测集的单个 TP/FP/FN 用户调节条件、权重或阈值。

每个候选版本相对基线比较：

- 评测集 Precision、Recall 和 $F_1$。
- 评测集 $TP$、$FP$ 和 $FN$ 数量。
- 预测用户数和预测用户比例。
- 校准集与评测集指标变化方向是否一致。
- 样本量允许时，规则影响在校准与评测间是否稳定。

只有候选版本满足约定业务约束、在独立评测集上改善主要指标且次要指标无实质回退时，才能通过。若已对同一个评测 CSV 重复评估多个版本，应将其标记为开发验证，并保留后续从未参与调优的标签批次作为最终验收集。

## 每轮必需交付物

- 版本化规则 YAML，例如 `rules_20260909_v1.yaml`。
- 版本化输出目录。
- 适用性报告。
- 候选规则证据表，包含被拒绝的候选。
- 包含校准与评测指标的基线-候选对比表。
- 明确结论：`promote`、`reject` 或 `needs more labeled data`。
- 简短原因说明，明确指出哪一项 YAML 改动带来了观察到的指标变化。

## 约束

- 保留用户已有改动，绝不覆盖历史批次输出目录。
- 每条规则都应有业务语言解释，并关联至结构化画像字段。
- 不得将年龄、性别、城市或原始文本作为自动低分结论。
- 业务预测输出不得暴露 $S_u$ 或个人数值风险分。
- 修改代码后运行聚焦测试；仅配置的实验至少要校验 YAML 解析并完成一次基线 CLI 运行。