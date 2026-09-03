from pathlib import Path
import json
import pandas as pd

from nps_analysis.pipeline import _read_sheet
from nps_analysis.potential_low import TEXT_COLUMNS, clean_text, SemanticMatcher, NEGATED

workbook = Path(r"D:\doc\nps\doc\指标汇总.xlsx")
matcher = SemanticMatcher("BAAI/bge-small-zh-v1.5", 0.82)

prototype_result = matcher.match("套餐价格太贵，收费不合理")
rows = []
for sheet in ["投诉明细", "触点轨迹"]:
    frame = _read_sheet(workbook, sheet).head(200)
    for idx, row in frame.iterrows():
        texts = []
        for col in TEXT_COLUMNS[sheet]:
            if col in frame.columns:
                text = clean_text(row.get(col))
                if text:
                    texts.append(text)
        if not texts:
            continue
        combined = " | ".join(texts)
        result = matcher.match(combined)
        rows.append(
            {
                "sheet": sheet,
                "row": int(idx),
                "text": combined[:180],
                "matched": result is not None,
                "matched_type": result[0] if result else None,
                "score": float(result[2]) if result else None,
                "negated": bool(NEGATED.search(combined)),
            }
        )

df = pd.DataFrame(rows)
out = {
    "prototype_result": prototype_result,
    "row_count": int(len(df)),
    "matched_count": int(df["matched"].sum()) if len(df) else 0,
    "top_rows": df.sort_values(["matched", "score"], ascending=[False, False]).head(20).to_dict(orient="records") if len(df) else [],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
