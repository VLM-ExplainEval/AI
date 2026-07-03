import csv
import os
import sys
from data_loader import load_grouped_data
from gemini_client import ask_gemini_order, parse_order
from metrics import get_gt, exact_match, calc_eta, calc_eta_simple
import time
from config import RESULT_DIR, TEST_JSON, TRAIN_JSON

GROUP = sys.argv[1] if len(sys.argv) > 1 else "low"
N_SAMPLES = 135  # 소규모 테스트

json_paths = [TEST_JSON, TRAIN_JSON]
samples = load_grouped_data(json_paths, group=GROUP, n=N_SAMPLES)

print(f"{GROUP} 그룹 {len(samples)}개 샘플로 실험 시작")

rows = []
org_scores = []
shuf_scores = []

for i, (video_id, frame_indices) in enumerate(samples):
    print(f"[{i+1}/{len(samples)}] {video_id} | 프레임: {frame_indices}")
    gt = get_gt(len(frame_indices))  # ['A', 'B', 'C']

    # Org
    try:
        response, _ = ask_gemini_order(video_id, frame_indices=frame_indices, shuffled=False)
        parsed = parse_order(response)
        org_correct = exact_match(gt, parsed)
        print(f"  Org EM: {org_correct}, 응답: {parsed}")
    except Exception as e:
        print(f"  Org 에러: {e}")
        org_correct = 0
        parsed = None

    # Shuf
    try:
        response_shuf, _ = ask_gemini_order(video_id, frame_indices=frame_indices, shuffled=True)
        parsed_shuf = parse_order(response_shuf)
        shuf_correct = exact_match(gt, parsed_shuf)  # Shuf 정답도 ['A','B','C']
        print(f"  Shuf EM: {shuf_correct}, 응답: {parsed_shuf}")
    except Exception as e:
        print(f"  Shuf 에러: {e}")
        shuf_correct = 0
        parsed_shuf = None

    org_scores.append(org_correct)
    shuf_scores.append(shuf_correct)
    rows.append({
        "video_id": video_id,
        "frame_indices": frame_indices,
        "org_em": org_correct,
        "shuf_em": shuf_correct,
        "org_pred": parsed,
        "shuf_pred": parsed_shuf,
    })

    time.sleep(15)

# 결과 계산
org_acc = sum(org_scores) / len(org_scores) * 100
shuf_acc = sum(shuf_scores) / len(shuf_scores) * 100
eta_vector = calc_eta(rows)
eta_simple = calc_eta_simple(org_acc, shuf_acc)

print(f"\n===== {GROUP} 그룹 결과 =====")
print(f"샘플 수: {len(samples)}")
print(f"Org EM: {org_acc:.2f}%")
print(f"Shuf EM: {shuf_acc:.2f}%")
print(f"η (VECTOR 공식): {eta_vector:.2f}%" if eta_vector is not None else "η (VECTOR 공식): 계산 불가")
print(f"η (단순 공식): {eta_simple:.2f}%" if eta_simple is not None else "η (단순 공식): 계산 불가")

from datetime import datetime
timestamp = datetime.now().strftime("%m%d_%H%M")
csv_path = os.path.join(RESULT_DIR, f"experiment1_{GROUP}_{timestamp}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"저장 완료: {csv_path}")