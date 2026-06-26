import csv
import os
from data_loader import load_test_data
from gemini_client import ask_gemini_order, parse_order
from metrics import get_gt, get_shuf_gt, exact_match, calc_eta
import time
from config import RESULT_DIR, NUM_FRAMES

data = load_test_data()

org_scores = []
shuf_scores = []
rows = []

for i, sample in enumerate(data):  # 안전 테스트: 3개만
    video_id = list(sample.keys())[0]
    info = sample[video_id]
    gt = get_gt(NUM_FRAMES)

    print(f"[{i+1}/3] {video_id}")

    # Org
    try:
        response, _ = ask_gemini_order(video_id, shuffled=False)
        parsed = parse_order(response)
        org_correct = exact_match(gt, parsed)
        print(f"  Org 응답: {parsed}, 정답: {gt}, EM: {org_correct}")
    except Exception as e:
        print(f"  Org 에러: {e}")
        org_correct = 0
        parsed = None

    # Shuf
    try:
        response_shuf, shuf_indices = ask_gemini_order(video_id, shuffled=True)
        parsed_shuf = parse_order(response_shuf)
        shuf_gt = get_shuf_gt(shuf_indices)
        shuf_correct = exact_match(shuf_gt, parsed_shuf)
        print(f"  Shuf 응답: {parsed_shuf}, 정답: {shuf_gt}, EM: {shuf_correct}")
    except Exception as e:
        print(f"  Shuf 에러: {e}")
        shuf_correct = 0
        parsed_shuf = None
        shuf_gt = None

    org_scores.append(org_correct)
    shuf_scores.append(shuf_correct)
    rows.append({
        "video_id": video_id,
        "org_em": org_correct,
        "shuf_em": shuf_correct,
        "org_pred": parsed,
        "shuf_pred": parsed_shuf,
        "shuf_gt": shuf_gt
    })

    time.sleep(7)

# 결과 계산
org_acc = sum(org_scores) / len(org_scores) * 100
shuf_acc = sum(shuf_scores) / len(shuf_scores) * 100
eta = calc_eta(org_acc, shuf_acc)

print(f"\n===== 최종 결과 =====")
print(f"Org 정확도: {org_acc:.2f}%")
print(f"Shuf 정확도: {shuf_acc:.2f}%")
if eta is not None:
    print(f"η: {eta:.2f}%")
else:
    print("η: 계산 불가 (Org 정확도 0)")

# CSV 저장
csv_path = os.path.join(RESULT_DIR, "experiment1_test3.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV 저장 완료: {csv_path}")