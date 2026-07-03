# 정답 순서 생성
def get_gt(num_frames):
    return [chr(65 + i) for i in range(num_frames)]  # ['A', 'B', 'C']

# 완전히 일치하면 1, 아니면 0
def exact_match(gt, pred):
    if pred is None:
        return 0
    return 1 if gt == pred else 0

# 부분적으로 맞은 비율
def partial_match(gt, pred):
    if pred is None:
        return 0
    correct = sum(1 for g, p in zip(gt, pred) if g == p)
    return correct / len(gt)

# η 계산 (VECTOR 논문 Eq.1)
def calc_eta(rows):
    denom = 0
    numer = 0
    for row in rows:
        org_em = int(row["org_em"])
        shuf_em = int(row["shuf_em"])
        if org_em == 1 and shuf_em == 0:
            denom += 1
            org_pred = row["org_pred"]
            shuf_pred = row["shuf_pred"]
            if org_pred is not None and shuf_pred is not None:
                if org_pred == shuf_pred:
                    numer += 1
    if denom == 0:
        return None
    return numer / denom * 100

# η 계산 (단순 공식)
def calc_eta_simple(org_acc, shuf_acc):
    if org_acc == 0:
        return None
    return (org_acc - shuf_acc) / org_acc * 100

def get_gt_from_order(order):
    """order[screen_pos] = 그 자리에 표시된 원본 프레임의 실제 시간순 인덱스"""
    n = len(order)
    labels = [chr(65 + i) for i in range(n)]
    frame_to_label = {orig: labels[screen] for screen, orig in enumerate(order)}
    return [frame_to_label[i] for i in range(n)]