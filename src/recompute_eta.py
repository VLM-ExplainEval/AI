# recompute_eta.py
import csv
import ast

def load_rows_from_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def parse_field(val):
                if val in ("None", "", None):
                    return None
                return ast.literal_eval(val)

            rows.append({
                "org_em": int(row["org_em"]),
                "shuf_em": int(row["shuf_em"]),
                "org_pred": parse_field(row["org_pred"]),
                "shuf_pred": parse_field(row["shuf_pred"]),
                "shuf_gt": parse_field(row["shuf_gt"]),
            })
    return rows


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from metrics import calc_eta

    csv_path = sys.argv[1]  # 예: results/experiment1_low_full.csv
    rows = load_rows_from_csv(csv_path)
    eta = calc_eta(rows)

    denom = sum(1 for r in rows if r["org_em"] == 1 and r["shuf_em"] == 0)
    print(f"파일: {csv_path}")
    print(f"전체 샘플 수: {len(rows)}")
    print(f"분모 (org 맞고 shuf 틀림): {denom}")
    print(f"η (VECTOR 공식): {eta:.2f}%" if eta is not None else "η: 계산 불가 (분모 0)")