from data_loader import load_test_data
from gemini_client import ask_gemini_order, parse_order
from metrics import get_gt, get_shuf_gt, exact_match, calc_eta
import time

data = load_test_data()

org_scores = []
shuf_scores = []

for i, sample in enumerate(data[:10]):
    video_id = list(sample.keys())[0]
    info = sample[video_id]
    gt = get_gt(len(info['sentences']))

    print(f"[{i+1}/10] {video_id}")

    # Org
    try:
        response, _ = ask_gemini_order(video_id, shuffled=False)
        parsed = parse_order(response)
        org_correct = exact_match(gt, parsed)
    except Exception as e:
        print(f"  Org 에러: {e}")
        org_correct = 0

    # Shuf
    try:
        response_shuf, shuf_indices = ask_gemini_order(video_id, shuffled=True)
        parsed_shuf = parse_order(response_shuf)
        shuf_gt = get_shuf_gt(shuf_indices)
        shuf_correct = exact_match(shuf_gt, parsed_shuf)
    except Exception as e:
        print(f"  Shuf 에러: {e}")
        shuf_correct = 0

    org_scores.append(org_correct)
    shuf_scores.append(shuf_correct)

    print(f"  Org EM: {org_correct}, Shuf EM: {shuf_correct}")
    time.sleep(7)

# 최종 결과
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