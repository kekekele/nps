# NPS 数据清洗与分析

本项目将 `doc/指标汇总.xlsx` 清洗为可直接读取的 CSV，并生成客群分布、评分分布、低分原因、时间轨迹和重要特征结果。

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

源文件中的手机号和宽带号已经脱敏。程序只对这些标识进行去空格、去无意义小数后缀等格式规范化，不再二次散列或加盐。

```powershell
.\.venv\Scripts\python.exe -m nps_analysis `
  --input "doc\指标汇总.xlsx" `
  --output "data" `
  --batch-id "20260901" `
  --version "v1"
```

## 输出

- `data/clean/{batch_id}/`：问卷、评分、子问题、客户声音、客户、家庭、月度指标和事件数据。
- `data/dictionary/`：评分题目和子问题选项字典。
- `data/analysis/{version}/`：分析样本、客群分布、评分分布、根因、轨迹和重要特征结果。
- `data/quality/{batch_id}/`：质量汇总与异常隔离记录。
- `data/manifest/{batch_id}.json`：输入哈希、输出文件行列数和文件哈希。

所有 CSV 使用 UTF-8 with BOM。客户和宽带关联键直接使用源文件中已脱敏标识的规范化值。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

当前实现只进行数据清洗和统计分析，不包含预测模型训练。

## 用户画像潜在低分规则优化

`nps_analysis.protential_low_score` 以 `user_profiles.jsonl` 和已测评用户标签为输入，不训练模型。它在标注集上自动从 YAML 中定义的离散规则权重与阈值中，按准确率、召回率或 $F_\beta$ 选择规则组合；独立评测集只计算最终指标，未测评用户只接受最终规则判断，不参与指标计算。

```powershell
.\.venv\Scripts\python.exe -m nps_analysis.protential_low_score.cli `
  --profiles "data\user_profiles.jsonl" `
  --labels "data\用户低分汇总.csv" `
  --evaluation-labels "data\用户低分评测集.csv" `
  --output "data\potential-low-score\20260907" `
  --mode balanced
```

`--labels` 是规则优化标注集；可选的 `--evaluation-labels` 是格式相同、且不参与优化的独立评测集。默认规则和时间窗口在 `nps_analysis/protential_low_score/rules.yaml` 中维护。输出的 `potential_low_user.csv` 只包含是否潜在低分、风险类型和规则原因；`rule_optimization.json` 保存优化集指标、独立评测指标（未传入评测集时为 `null`），以及选中的内部权重和阈值。

## 潜在低分规则识别

规则引擎独立运行，不会触发现有全量清洗分析流程。`--as-of` 是证据截止日，规则只读取该日期之前的行为：

```powershell
.\.venv\Scripts\python.exe -m nps_analysis.potential_low_cli `
  --input "doc\指标汇总.xlsx" `
  --output "data\potential-low\20260901" `
  --as-of "2026-09-01" `
  --rule-version "potential-low-v3" `
  --potential-low-min-types 3 `
  --medium-priority-min-types 2
```

输出包括用户结果、用户类型、证据、类型画像、问卷级匹配明细、匹配指标汇总、规则字典、批次质量和清单。其中 `rule_validation_detail.csv`逐条标记 TP/FP/FN/TN，`rule_validation.csv`输出总体、分月和分风险类型的 precision、recall、F1、Jaccard、specificity、accuracy 与 lift。文本规则只读取投诉与触点的实际文本字段；业务订购、资费变更、限速加包等只生成结构化证据。

### 可选中文语义匹配

默认只运行可审计的结构化与正则规则。需要补充同义改写的文本召回时，可启用本地句向量匹配；首次运行会下载指定模型，之后模型从本地缓存加载，投诉文本不会发送到外部服务：

```powershell
.\.venv\Scripts\python.exe -m nps_analysis.potential_low_cli `
  --input "doc\指标汇总.xlsx" `
  --output "data\potential-low\20260902_semantic" `
  --as-of "2026-09-02" `
  --rule-version "potential-low-semantic-v1" `
  --semantic-enabled `
  --semantic-model "BAAI/bge-small-zh-v1.5" `
  --semantic-threshold 0.82
```

语义命中会输出为 `SEMANTIC_*` 中等证据，并记录最高余弦相似度和命中的原型句。它只补充投诉未解决、资费不满、办理受阻、营销争议、提醒不足五类文本证据；不会单独触发流量超套或家宽体验类型，且仍受已解决文本、超套和狼号等既有排除条件约束。应使用独立输出目录，并与纯规则版本在历史问卷上对比后冻结阈值。