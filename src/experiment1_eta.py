import csv
import os
from data_loader import load_grouped_data, load_images_as_base64
from gemini_client import ask_gemini_order, parse_order
from metrics import get_gt, get_shuf_gt, exact_match, calc_eta
import time
from config import RESULT_DIR

GROUP = "low"   # "low" 또는 "high" 로 바꿔서 실행
N_SAMPLES = 10  # 소규모 테스트

json_paths = ["data/test.json", "data/train.json"]
samples = load_grouped_data(json_paths, group=GROUP, n=N_SAMPLES)

print(f"{GROUP} 그룹 {len(samples)}개 샘플로 테스트 시작")

org_scores = []
shuf_scores = []
rows = []

for i, (video_id, frame_indices) in enumerate(samples):
    print(f"[{i+1}/{len(samples)}] {video_id} | 프레임: {frame_indices}")
    gt = list(range(3))

    try:
        response, _ = ask_gemini_order(video_id, shuffled=False, frame_indices=frame_indices)
        parsed = parse_order(response)
        org_correct = exact_match(gt, parsed)
        print(f"  Org EM: {org_correct}, 응답: {parsed}")
    except Exception as e:
        print(f"  Org 에러: {e}")
        org_correct = 0
        parsed = None

    try:
        response_shuf, shuf_indices = ask_gemini_order(video_id, shuffled=True, frame_indices=frame_indices)
        parsed_shuf = parse_order(response_shuf)
        shuf_gt = get_shuf_gt(shuf_indices)
        shuf_correct = exact_match(shuf_gt, parsed_shuf)
        print(f"  Shuf EM: {shuf_correct}, 응답: {parsed_shuf}")
    except Exception as e:
        print(f"  Shuf 에러: {e}")
        shuf_correct = 0
        parsed_shuf = None
        shuf_gt = None

    org_scores.append(org_correct)
    shuf_scores.append(shuf_correct)
    rows.append({
        "video_id": video_id,
        "frame_indices": frame_indices,
        "org_em": org_correct,
        "shuf_em": shuf_correct,
        "org_pred": parsed,
        "shuf_pred": parsed_shuf,
        "shuf_gt": shuf_gt
    })

    time.sleep(15)

org_acc = sum(org_scores) / len(org_scores) * 100
shuf_acc = sum(shuf_scores) / len(shuf_scores) * 100
eta = calc_eta(org_acc, shuf_acc)

print(f"\n===== {GROUP} 그룹 결과 =====")
print(f"Org: {org_acc:.2f}%, Shuf: {shuf_acc:.2f}%")
print(f"η: {eta:.2f}%" if eta else "η: 계산 불가")

csv_path = os.path.join(RESULT_DIR, f"experiment1_{GROUP}_test.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"저장 완료: {csv_path}")