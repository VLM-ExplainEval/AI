# 정답 순서 생성
def get_gt(num_frames):
    return list(range(num_frames))

# 섞인 순서 기준 정답 계산
def get_shuf_gt(shuf_indices):
    gt_shuf = [0] * len(shuf_indices)
    for new_pos, original_idx in enumerate(shuf_indices):
        gt_shuf[original_idx] = new_pos
    return gt_shuf

# 완전히 일치하면 1, 아니면 0
def exact_match(gt, pred):
    return 1 if gt == pred else 0

# 부분적으로 맞은 비율
def partial_match(gt, pred):
    if pred is None:
        return 0
    correct = sum(1 for g, p in zip(gt, pred) if g == p)
    return correct / len(gt)

# η 계산
def calc_eta(org_score, shuf_score):
    if org_score == 0:
        return None
    return (org_score - shuf_score) / org_score * 100