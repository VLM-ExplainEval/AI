"""
실험2 - 채점(Judge) 전용 모듈.

역할: 이미 생성된 설명 텍스트(CSV 또는 dict)를 받아 품질 점수를 매긴다.
생성(Generation)과 완전히 분리되어 있어, 어떤 모델이 만든 설명이든(Gemini/Qwen)
CSV만 있으면 채점할 수 있다. judge_model 인자로 채점자도 바꿀 수 있다.

주의 (중요): 기존 judge_explanations()는 이미지를 넘기지 않고 텍스트(Existence, CoT)만으로
채점했다. 이는 "시각적 그라운딩" 점수가 실제 화면과의 일치가 아니라 텍스트 간 어휘 중첩을
보는 구조였을 가능성이 있다. 아래에 이미지 포함 버전(judge_explanations_with_images)을
같이 만들어두었으니, 두 방식을 비교해보는 것을 권장한다.
"""

import json
from gemini_client import call_gemini_with_retry
from data_loader import load_images_as_base64


def _build_judge_prompt(explanations, existence_texts, cot_texts):
    return f"""다음 설명들을 엄격하게 평가해줘.

정답 시각적 요소 (Existence):
{chr(10).join(existence_texts)}

정답 인과 설명 (CoT):
{chr(10).join(cot_texts)}

모델이 생성한 설명:
- 논리적 근거: {explanations.get('logical', '')}
- 시각적 그라운딩: {explanations.get('visual', '')}
- 인과적 설명: {explanations.get('causal', '')}
- 대조적 설명: {explanations.get('contrastive', '')}

채점 기준 (엄격하게):
- logical_score: 논리적 흐름이 명확한가?
  1=근거없음, 2=매우일반적, 3=보통, 4=명확, 5=매우구체적
- visual_score: Existence에 나열된 구체적 요소(색상/객체/위치)를 실제로 언급했는가?
  1=Existence 요소 전혀 언급안함
  2=배경(실내/실외)만 언급
  3=일반적 객체만 언급(사람, 물건 등)
  4=Existence 요소 일부 구체적 언급
  5=Existence 요소 대부분 구체적으로 언급
- causal_score: CoT의 인과관계를 실제로 설명했는가?
  1=인과관계 없음, 3=일반적 인과, 5=CoT와 유사한 구체적 인과
- contrastive_score: 다른 순서가 왜 틀렸는지 구체적으로 반박했는가?
  1=반박없음, 3=일반적반박, 5=구체적이고납득가능한반박

JSON으로만 답해줘:
{{"logical_score": 점수, "visual_score": 점수, "causal_score": 점수, "contrastive_score": 점수}}"""


def _parse_judge_response(text):
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"logical_score": None, "visual_score": None,
                "causal_score": None, "contrastive_score": None}


def _get_existence_cot_texts(frame_indices, video_id, all_data):
    info = all_data.get(video_id, {})
    existence = info.get('Existence', [])
    cot = info.get('cot', [])

    existence_texts = []
    cot_texts = []
    labels = ['A', 'B', 'C']
    for j, idx in enumerate(frame_indices):
        if idx < len(existence):
            existence_texts.append(f"이벤트 {labels[j]}: {existence[idx]}")
        if idx < len(cot):
            cot_texts.append(f"이벤트 {labels[j]}: {cot[idx]}")
    return existence_texts, cot_texts


def _is_empty_explanations(explanations):
    """설명이 전부 None이거나 빈 문자열이면 True (채점 스킵 대상)"""
    values = [explanations.get(k) for k in ("logical", "visual", "causal", "contrastive")]
    return all(v is None or (isinstance(v, str) and v.strip() == "") for v in values)


_EMPTY_SCORE = {"logical_score": None, "visual_score": None,
                "causal_score": None, "contrastive_score": None}


def judge_explanations(explanations, frame_indices, video_id, all_data, judge_model="gemini"):
    """
    기존 방식과 동일: 텍스트(Existence, CoT)만으로 채점. 이미지는 사용하지 않는다.
    """
    if _is_empty_explanations(explanations):
        print(f"  [채점 스킵] {video_id}: 생성된 설명이 없어(파싱 실패 등) 채점하지 않음")
        return dict(_EMPTY_SCORE)

    if judge_model == "qwen":
        from qwen_client import judge_explanations_qwen
        return judge_explanations_qwen(explanations, frame_indices, video_id, all_data)

    if judge_model != "gemini":
        raise NotImplementedError(f"judge_model={judge_model} 은 아직 구현되지 않았습니다.")

    existence_texts, cot_texts = _get_existence_cot_texts(frame_indices, video_id, all_data)
    prompt = _build_judge_prompt(explanations, existence_texts, cot_texts)

    try:
        text = call_gemini_with_retry([{"text": prompt}])
        return _parse_judge_response(text)
    except Exception as e:
        print(f"  [채점 에러] {video_id}: {e}")
        return {"logical_score": None, "visual_score": None,
                "causal_score": None, "contrastive_score": None}


def judge_explanations_with_images(explanations, frame_indices, video_id, all_data, shuffled, judge_model="gemini"):
    """
    이미지 포함 채점. "시각적 그라운딩" 점수를 실제 화면 근거로 매기고 싶을 때 사용.

    주의: shuffled 값에 따라 Judge에게 보여줄 이미지 순서가 달라져야 함
    (설명이 생성된 조건과 동일한 이미지를 보여줘야 공정한 채점이 됨).
    """
    if _is_empty_explanations(explanations):
        print(f"  [채점 스킵] {video_id}: 생성된 설명이 없어(파싱 실패 등) 채점하지 않음")
        return dict(_EMPTY_SCORE)

    if judge_model == "qwen":
        from qwen_client import judge_explanations_with_images_qwen
        return judge_explanations_with_images_qwen(explanations, frame_indices, video_id, all_data, shuffled)

    if judge_model != "gemini":
        raise NotImplementedError(f"judge_model={judge_model} 은 아직 구현되지 않았습니다.")

    images, order = load_images_as_base64(video_id, frame_indices, shuffled=shuffled)
    existence_texts, cot_texts = _get_existence_cot_texts(frame_indices, video_id, all_data)
    labels = [chr(65 + i) for i in range(len(frame_indices))]

    contents = []
    for k in range(len(frame_indices)):
        contents.append({"text": f"Frame {labels[k]}:"})
        contents.append({"inline_data": {"mime_type": "image/jpeg", "data": images[k]}})

    prompt = _build_judge_prompt(explanations, existence_texts, cot_texts)
    contents.append({"text": prompt})

    try:
        text = call_gemini_with_retry(contents)
        return _parse_judge_response(text)
    except Exception as e:
        print(f"  [이미지 포함 채점 에러] {video_id}: {e}")
        return {"logical_score": None, "visual_score": None,
                "causal_score": None, "contrastive_score": None}