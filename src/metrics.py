def get_gt(num_frames):
    return list(range(num_frames))

def get_shuf_gt(shuf_indices):
    gt_shuf = [0] * len(shuf_indices)
    for new_pos, original_idx in enumerate(shuf_indices):
        gt_shuf[original_idx] = new_pos
    return gt_shuf

def exact_match(gt, pred):
    return 1 if gt == pred else 0

def partial_match(gt, pred):
    if pred is None:
        return 0
    correct = sum(1 for g, p in zip(gt, pred) if g == p)
    return correct / len(gt)

def calc_eta(org_score, shuf_score):
    if org_score == 0:
        return None
    return (org_score - shuf_score) / org_score * 100