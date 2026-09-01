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