# main 파일 실행 
from data_loader import load_test_data
from gemini_client import ask_gemini_order, parse_order
from metrics import get_gt, get_shuf_gt, exact_match, calc_eta

data = load_test_data()

# 샘플 1개 테스트
sample = data[0]
video_id = list(sample.keys())[0]
info = sample[video_id]
gt = get_gt(len(info['sentences']))

print(f"영상 ID: {video_id}")

# Org
print("\nOrg 테스트 중...")
response, _ = ask_gemini_order(video_id, shuffled=False)
parsed = parse_order(response)
org_correct = exact_match(gt, parsed)
print(f"정답: {gt}")
print(f"예측: {parsed}")
print(f"EM: {org_correct}")

# Shuf
print("\nShuf 테스트 중...")
response_shuf, shuf_indices = ask_gemini_order(video_id, shuffled=True)
parsed_shuf = parse_order(response_shuf)
shuf_gt = get_shuf_gt(shuf_indices)
shuf_correct = exact_match(shuf_gt, parsed_shuf)
print(f"정답: {shuf_gt}")
print(f"예측: {parsed_shuf}")
print(f"EM: {shuf_correct}")

# η
eta = calc_eta(org_correct, shuf_correct)
print(f"\nη: {eta}%" if eta is not None else "\nη: 계산 불가 (Org도 틀림)")