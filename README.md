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

## 潜在低分规则识别

规则引擎独立运行，不会触发现有全量清洗分析流程。`--as-of` 是证据截止日，规则只读取该日期之前的行为：

```powershell
.\.venv\Scripts\python.exe -m nps_analysis.potential_low_cli `
  --input "doc\指标汇总.xlsx" `
  --output "data\potential-low\20260901" `
  --as-of "2026-09-01" `
  --rule-version "potential-low-v1"
```

输出包括用户结果、用户类型、证据、类型画像、问卷级匹配明细、匹配指标汇总、规则字典、批次质量和清单。其中 `rule_validation_detail.csv`逐条标记 TP/FP/FN/TN，`rule_validation.csv`输出总体、分月和分风险类型的 precision、recall、F1、Jaccard、specificity、accuracy 与 lift。文本规则只读取投诉与触点的实际文本字段；业务订购、资费变更、限速加包等只生成结构化证据。