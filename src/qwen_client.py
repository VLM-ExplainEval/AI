"""
Qwen2-VL 클라이언트. gemini_client.py와 완전히 동일한 인터페이스를 제공한다.
- ask_qwen_order(video_id, frame_indices, shuffled) -> (raw_text, order)
- parse_qwen_order(raw_text) -> list 또는 None   (gemini_client.parse_order와 동일 로직 재사용)

experiment1_eta.py, run_experiment2.py에서 모델 선택 시
'from qwen_client import ask_qwen_order as ask_gemini_order, parse_qwen_order as parse_order'
식으로 바꿔 끼우면 기존 코드를 그대로 재사용할 수 있다.

주의: 아직 검증 안 된 부분 (실제로 서버에서 돌려보기 전까지는 가정)
- Qwen이 Gemini와 동일하게 "[...]" 형태의 리스트 문자열로 응답한다는 보장은 없음
  → parse_qwen_order()에서 우선 parse_order와 동일한 정규식을 재사용하되,
    실제 응답을 보고 Qwen 전용 파싱이 필요하면 그때 수정해야 함
- max_new_tokens, 프롬프트 문구는 Gemini와 최대한 동일하게 맞췄으나
  모델별로 반응이 다를 수 있어 소규모 테스트 후 조정 필요
"""

import re
import gc
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from data_loader import load_images_as_base64, get_image_folder, get_filename

QWEN_MODEL_PATH = "./qwen_model_2b"  # 7B가 TITAN Xp 환경에서 불안정하여 2B로 전환

_model = None
_processor = None


def _load_model():
    """
    모델은 한 번만 로드해서 재사용 (매 호출마다 로드하면 너무 느림)

    device_map="auto"만 쓰면 GPU 간 분배가 불균등해서 특정 GPU에서 OOM이 날 수 있음
    (실제로 GPU 0에 10GB+ 몰리는 현상 확인됨). max_memory로 GPU당 상한을 강제 지정해
    강제로 고르게 분산시킨다.
    """
    global _model, _processor
    if _model is None:
        print("Qwen2-VL 모델 로드 중...")

        import torch as _torch
        num_gpus = _torch.cuda.device_count()
        # 2B 모델은 GPU 1장(6~8GB)이면 충분하지만, 안전하게 여러 장에 나눠지도록 여유 있게 지정
        max_memory = {i: "10GiB" for i in range(num_gpus)}

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
        )
        _processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_PATH,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,  # 이미지 해상도 상한 강제 -> 특정 이미지가 커서 메모리 폭증하는 것 방지
        )
        print("Qwen2-VL 모델 로드 완료")
    return _model, _processor


def _generate(messages, max_new_tokens=256):
    model, processor = _load_model()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 재현성을 위해 그리디 디코딩으로 고정 (랜덤 샘플링 방지)
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 다음 호출에서 메모리가 계속 쌓이는 것을 막기 위한 명시적 정리
    del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return output_text


def _build_content(video_id, frame_indices, shuffled):
    """실험1/실험2 공용: 이미지 content 리스트와 order를 만들어 반환"""
    n = len(frame_indices)
    images, order = load_images_as_base64(video_id, frame_indices, shuffled=shuffled)
    folder = get_image_folder(video_id)
    labels = [chr(65 + i) for i in range(n)]

    content = []
    for k, rel_pos in enumerate(order):
        actual_frame = frame_indices[rel_pos]
        filename = get_filename(folder, actual_frame)
        image_path = f"{folder}/{filename}"
        content.append({"type": "text", "text": f"Frame {labels[k]}:"})
        content.append({"type": "image", "image": image_path})

    return content, order, labels, n


def ask_qwen_order(video_id, frame_indices, shuffled=False):
    """
    gemini_client.ask_gemini_order와 동일한 시그니처/반환값.
    실험1용: 이미지만 보여주고 순서(라벨 리스트)만 요청.
    """
    content, order, labels, n = _build_content(video_id, frame_indices, shuffled)

    content.append({
        "type": "text",
        "text": (
            f"These are {n} frames shown above, labeled {labels} in the order shown.\n"
            f"Arrange them in the order these events actually occur.\n"
            f"Reply with ONLY a Python list like {labels}. No explanation."
        ),
    })

    messages = [{"role": "user", "content": content}]
    raw_text = _generate(messages, max_new_tokens=64)
    return raw_text, order


def ask_qwen_order_with_explanation(video_id, frame_indices, shuffled=False):
    """
    gemini_client.ask_gemini_order_with_explanation과 동일한 시그니처/반환값.
    실험2용: 순서 예측 + 4가지 설명(logical/visual/causal/contrastive)을 생성.

    설계 변경 이유: Qwen2-VL-2B에게 4개 필드를 한 번에 JSON으로 요청하면
    (1) 응답이 길어져 토큰 제한에 걸려 잘리고, (2) 작은따옴표/개행 미이스케이프 등
    JSON 문법을 안정적으로 못 지키는 문제가 실측 확인되었다.
    프롬프트 제약(문장 수 강제 등)으로 응답을 인위적으로 짧게 깎는 것은 모델의
    자연스러운 응답 품질을 왜곡시켜 Gemini와의 공정한 비교를 해치므로,
    대신 순서 예측 1회 + 설명 4개를 각각 개별 호출로 나눠 받는다.
    시간은 더 걸리지만 모델의 응답 자체는 손대지 않는다.
    """
    # 1. 순서 예측 (기존과 동일한 간단한 요청)
    order_raw, order = ask_qwen_order(video_id, frame_indices, shuffled=shuffled)
    pred = parse_qwen_order(order_raw)

    # 2. 설명 4개를 각각 개별 질문으로 요청 (JSON 강제 없음, 자유 텍스트 그대로 받음)
    explanation_prompts = {
        "logical": "이 프레임들이 보여주는 순서가 왜 논리적으로 맞는지 설명해줘.",
        "visual": "각 프레임에서 어떤 시각적 요소(색상/동작/객체)가 이 순서를 뒷받침하는지 설명해줘.",
        "causal": "이 이벤트들 사이에 어떤 인과관계가 있는지 설명해줘.",
        "contrastive": "만약 순서가 다르다면 왜 말이 안 되는지 설명해줘.",
    }

    result = {"order": pred}
    for key, question in explanation_prompts.items():
        content, _, labels, n = _build_content(video_id, frame_indices, shuffled)
        content.append({
            "type": "text",
            "text": (
                f"These are {n} frames from a video, shown above labeled {labels} in the order shown.\n"
                f"방금 판단한 순서는 {pred if pred else labels} 이다. 이를 바탕으로:\n"
                f"{question}"
            ),
        })
        messages = [{"role": "user", "content": content}]
        try:
            answer = _generate(messages, max_new_tokens=256)
            answer = _strip_repeated_question(answer, question)
            result[key] = answer
        except Exception as e:
            print(f"    [Qwen {key} 생성 에러] {e}")
            result[key] = None

    return result, order


def _strip_repeated_question(answer, question):
    """
    Qwen2-VL-2B가 질문을 그대로(또는 문장부호만 살짝 바꿔서) 반복하고 나서 답하는
    습관이 있어(실측 확인됨). 이건 모델의 실제 판단 내용이 아니라 프롬프트가
    그대로 메아리친 것이므로, 응답 내용 자체를 바꾸지 않는 선에서 이 반복된
    질문 부분만 제거한다.

    문장부호(마침표 vs 콜론 등) 차이 때문에 정확 일치 비교가 실패하는 경우가
    실측됐으므로, 문장부호/공백을 제거한 뒤 비교한다.
    """
    import re as _re

    def _normalize(s):
        # 끝의 문장부호(. : ! ? 및 공백)를 제거하고 비교
        return _re.sub(r'[.:!?\s]+$', '', s.strip())

    answer = answer.strip()
    q_norm = _normalize(question)

    # answer 앞부분에서 question과 같은 길이만큼 잘라 정규화 비교
    # (정확한 길이를 모르므로, question 길이 근방에서 실제 구분자(: 또는 줄바꿈)를 찾음
    for sep in ['\n\n', '\n', ':']:
        idx = answer.find(sep)
        if idx != -1:
            candidate = answer[:idx]
            if _normalize(candidate) == q_norm:
                return answer[idx + len(sep):].strip()

    # 위에서 못 찾았으면 전체 첫 줄 기준으로도 한 번 더 시도
    first_line, _, rest = answer.partition('\n')
    if _normalize(first_line) == q_norm and rest:
        return rest.strip()

    return answer


def parse_qwen_order(response_text):
    """
    gemini_client.parse_order와 동일 로직.
    Qwen 응답 스타일을 실제로 확인한 뒤, 다르면 이 함수만 수정하면 됨.
    """
    if response_text is None:
        return None
    match = re.search(r'\[[^\]]+\]', response_text)
    if match:
        try:
            result = eval(match.group())
            if isinstance(result, list):
                return result
        except Exception:
            return None
    return None


def _judge_via_qwen(prompt_text, image_paths=None):
    """
    explain_judge.py의 _build_judge_prompt()가 만든 텍스트 프롬프트를
    Qwen에 넣어 답을 받고 JSON으로 파싱한다. image_paths가 주어지면 이미지도 같이 넣는다.

    Qwen2-VL-2B가 근거 없이 곧바로 만점(5점)만 뱉는 self-judge 편향이 실측 확인되었다
    (Gemini는 프롬프트 고도화로 이 편향을 줄였으나, 동일 프롬프트가 Qwen에는 효과가 없었음).
    이를 완화하기 위해 "각 항목마다 근거를 한 문장으로 먼저 쓰고, 그 다음 점수를 매기라"는
    지시를 추가한다. 근거를 먼저 쓰게 하면 점수를 아무 생각 없이 찍는 것을 억제하는 효과를 기대함.
    """
    guard_instruction = (
        "\n\n중요: 점수를 매기기 전에, 각 항목마다 왜 그 점수를 주는지 근거를 한 문장으로 먼저 써라. "
        "무조건 높은 점수(5점)를 주지 말고, 채점 기준에 명시된 조건을 실제로 충족하는지 엄격하게 확인해라. "
        "근거 서술 후, 마지막 줄에 JSON만 따로 출력해라."
    )
    full_prompt = prompt_text + guard_instruction

    content = []
    if image_paths:
        labels = [chr(65 + i) for i in range(len(image_paths))]
        for k, path in enumerate(image_paths):
            content.append({"type": "text", "text": f"Frame {labels[k]}:"})
            content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": full_prompt})

    messages = [{"role": "user", "content": content}]
    # 근거 서술까지 포함해야 하므로 토큰 여유를 늘림 (기존 256 -> 512)
    raw_text = _generate(messages, max_new_tokens=512)
    return raw_text


def _parse_judge_response_qwen(text):
    """
    explain_judge._parse_judge_response와 달리, Qwen은 근거 서술 뒤에 JSON이 오므로
    텍스트 전체에서 마지막 {...} 블록을 찾아 파싱한다.
    """
    import json as _json
    import re as _re

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    # 근거 서술 뒤에 JSON이 붙는 경우를 위해, 텍스트 내 마지막 {...} 블록을 찾음
    matches = _re.findall(r'\{[^{}]*\}', text, flags=_re.DOTALL)
    for candidate in reversed(matches):
        try:
            return _json.loads(candidate)
        except Exception:
            continue

    # 못 찾았으면 기존 방식(전체를 그대로 파싱) 시도
    try:
        return _json.loads(text.strip())
    except Exception:
        return {"logical_score": None, "visual_score": None,
                "causal_score": None, "contrastive_score": None}


def judge_explanations_qwen(explanations, frame_indices, video_id, all_data):
    """
    explain_judge.judge_explanations와 동일한 역할(텍스트만, self-judge).
    explain_judge.py의 프롬프트 생성 함수는 재사용하되, 파싱은 Qwen 전용 함수를 쓴다
    (근거 서술이 앞에 붙는 응답 스타일에 맞춰야 하므로).
    """
    from explain_judge import _build_judge_prompt, \
        _get_existence_cot_texts, _is_empty_explanations, _EMPTY_SCORE

    if _is_empty_explanations(explanations):
        print(f"  [채점 스킵] {video_id}: 생성된 설명이 없어(파싱 실패 등) 채점하지 않음")
        return dict(_EMPTY_SCORE)

    existence_texts, cot_texts = _get_existence_cot_texts(frame_indices, video_id, all_data)
    prompt = _build_judge_prompt(explanations, existence_texts, cot_texts)

    try:
        raw_text = _judge_via_qwen(prompt)
        print(f"    [Qwen 채점 원본 응답] {video_id}: {raw_text[:300]}")
        return _parse_judge_response_qwen(raw_text)
    except Exception as e:
        print(f"  [Qwen 채점 에러] {video_id}: {e}")
        return dict(_EMPTY_SCORE)


def judge_explanations_with_images_qwen(explanations, frame_indices, video_id, all_data, shuffled):
    """
    explain_judge.judge_explanations_with_images와 동일한 역할(이미지 포함, self-judge).
    """
    from explain_judge import _build_judge_prompt, \
        _get_existence_cot_texts, _is_empty_explanations, _EMPTY_SCORE

    if _is_empty_explanations(explanations):
        print(f"  [채점 스킵] {video_id}: 생성된 설명이 없어(파싱 실패 등) 채점하지 않음")
        return dict(_EMPTY_SCORE)

    folder = get_image_folder(video_id)
    order = list(range(len(frame_indices)))
    if shuffled:
        import random as _random
        _random.shuffle(order)
    image_paths = []
    for rel_pos in order:
        actual_frame = frame_indices[rel_pos]
        filename = get_filename(folder, actual_frame)
        image_paths.append(f"{folder}/{filename}")

    existence_texts, cot_texts = _get_existence_cot_texts(frame_indices, video_id, all_data)
    prompt = _build_judge_prompt(explanations, existence_texts, cot_texts)

    try:
        raw_text = _judge_via_qwen(prompt, image_paths=image_paths)
        print(f"    [Qwen 이미지 포함 채점 원본 응답] {video_id}: {raw_text[:300]}")
        return _parse_judge_response_qwen(raw_text)
    except Exception as e:
        print(f"  [Qwen 이미지 포함 채점 에러] {video_id}: {e}")
        return dict(_EMPTY_SCORE)