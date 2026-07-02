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
    if pred is None:
        return 0
    return 1 if gt == pred else 0

# 부분적으로 맞은 비율
def partial_match(gt, pred):
    if pred is None:
        return 0
    correct = sum(1 for g, p in zip(gt, pred) if g == p)
    return correct / len(gt)

# η 계산 (VECTOR 논문 Eq.1 기준)
# η = (Org 맞고 Shuf 틀린 영상들 중, org_pred == shuf_pred(원본 identity 기준)인 비율)
def calc_eta(rows):
    """
    rows: 각 영상마다 아래 키를 가진 딕셔너리 리스트
        - org_em: int (0 or 1)
        - shuf_em: int (0 or 1)
        - org_pred: list[int] or None
        - shuf_pred: list[int] or None
        - shuf_gt: list[int] or None
    """
    denom = 0
    numer = 0

    for row in rows:
        org_em = row["org_em"]
        shuf_em = row["shuf_em"]

        # 분모 조건: Org는 맞고 Shuf는 틀린 경우만
        if org_em == 1 and shuf_em == 0:
            denom += 1

            org_pred = row["org_pred"]
            shuf_pred = row["shuf_pred"]
            shuf_gt = row["shuf_gt"]

            if org_pred is None or shuf_pred is None or shuf_gt is None:
                continue  # 파싱 실패 -> 불일치로 처리 (numer 증가 안 함)

            n = len(shuf_gt)
            try:
                # shuf_pred(셔플 화면 기준 new_pos)를 원본 프레임 identity 기준으로 재정렬
                remapped_shuf_pred = [shuf_pred[shuf_gt[i]] for i in range(n)]
            except (IndexError, TypeError):
                continue

            if org_pred == remapped_shuf_pred:
                numer += 1

    if denom == 0:
        return None
    return numer / denom * 100