"""
실험2 - 생성(Generation) 전용 모듈.

역할: 이미지+프레임을 받아 4가지 설명(logical/visual/causal/contrastive)과 순서 예측을 생성한다.
채점(Judge)과 완전히 분리되어 있어, 이 모듈만으로 CSV(설명 텍스트)를 만들 수 있다.

나중에 Qwen2-VL을 붙일 때는 이 파일에 generate_explanations_qwen() 같은 함수를 추가하고,
run_experiment2.py에서 model 인자로 선택하게 하면 된다.
"""

import time
from gemini_client import ask_gemini_order_with_explanation
from metrics import get_gt_from_order, exact_match


def generate_explanations(video_id, frame_indices, shuffled, model="gemini"):
    """
    한 샘플(video_id, frame_indices)에 대해 Org 또는 Shuf 조건으로 설명+순서를 생성.

    Args:
        video_id: 영상 ID
        frame_indices: 사용할 프레임 인덱스 리스트 (예: [0, 2, 3])
        shuffled: True면 Shuf 조건, False면 Org 조건
        model: "gemini" (현재 유일하게 구현됨). 추후 "qwen" 추가 예정.

    Returns:
        dict: {
            "video_id": str,
            "shuffled": bool,
            "order": list,          # 실제 화면에 표시된 원본 인덱스 순서
            "pred": list or None,   # 모델이 예측한 라벨 순서 (예: ['A','B','C'])
            "em": int,               # exact match (0 또는 1)
            "explanations": {
                "logical": str, "visual": str, "causal": str, "contrastive": str
            }
        }
    """
    if model != "gemini":
        raise NotImplementedError(f"model={model} 은 아직 구현되지 않았습니다.")

    try:
        result, order = ask_gemini_order_with_explanation(
            video_id, frame_indices, shuffled=shuffled
        )
        pred = result.get("order")
        gt = get_gt_from_order(order)
        em = exact_match(gt, pred)
        explanations = {
            "logical": result.get("logical"),
            "visual": result.get("visual"),
            "causal": result.get("causal"),
            "contrastive": result.get("contrastive"),
        }
    except Exception as e:
        print(f"  [생성 에러] {video_id} (shuffled={shuffled}): {e}")
        order = None
        pred = None
        em = 0
        explanations = {"logical": None, "visual": None, "causal": None, "contrastive": None}

    return {
        "video_id": video_id,
        "shuffled": shuffled,
        "order": order,
        "pred": pred,
        "em": em,
        "explanations": explanations,
    }


def generate_org_and_shuf(video_id, frame_indices, model="gemini", sleep_between=5):
    """
    한 샘플에 대해 Org, Shuf 둘 다 생성. (기존 코드의 순차 호출 로직과 동일한 순서 유지)
    """
    org_result = generate_explanations(video_id, frame_indices, shuffled=False, model=model)
    time.sleep(sleep_between)
    shuf_result = generate_explanations(video_id, frame_indices, shuffled=True, model=model)
    return org_result, shuf_result
