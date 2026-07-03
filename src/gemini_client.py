import os
import re
import time
import random
import json
from dotenv import load_dotenv
from google import genai
from config import MODEL_NAME, TEST_JSON, TRAIN_JSON
from data_loader import load_images_as_base64, load_json_files

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_sentences(video_id):
    all_data = load_json_files([TEST_JSON, TRAIN_JSON])
    for sample in all_data:
        vid = list(sample.keys())[0]
        if vid == video_id:
            return sample[vid]['sentences']
    return None

def call_gemini_with_retry(contents, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "500" in str(e):
                wait = 45
                print(f"  재시도 중... {wait}초 대기 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e
    raise Exception("최대 재시도 횟수 초과")

def ask_gemini_order(video_id, frame_indices, shuffled=False, max_retries=5):
    n = len(frame_indices)
    images, order = load_images_as_base64(video_id, frame_indices, shuffled=shuffled)
    labels = [chr(65 + i) for i in range(n)]

    contents = []
    for k in range(n):
        contents.append({"text": f"Frame {labels[k]}:"})
        contents.append({
            "inline_data": {"mime_type": "image/jpeg", "data": images[k]}
        })
    contents.append({
        "text": f"These are {n} frames shown above, labeled {labels} in the order shown.\n"
                f"Arrange them in the order these events actually occur.\n"
                f"Reply with ONLY a Python list like {labels}. No explanation."
    })
    return call_gemini_with_retry(contents), order

def ask_gemini_order_with_explanation(video_id, frame_indices, shuffled=False):
    """실험 2: 이미지만으로 순서 예측 + 4가지 설명 생성 (캡션 없음)"""
    n = len(frame_indices)
    images, order = load_images_as_base64(video_id, frame_indices, shuffled)
    labels = [chr(65 + i) for i in range(n)]

    contents = []
    for k in range(n):
        contents.append({"text": f"Frame {labels[k]}:"})
        contents.append({
            "inline_data": {"mime_type": "image/jpeg", "data": images[k]}
        })

    contents.append({
        "text": f"These are {n} frames from a video, shown above labeled {labels} in the order shown.\n"
                f"Based ONLY on the visual content, determine the order these events actually occurred, "
                f"and explain why in 4 ways.\n\n"
                f"아래 JSON 형식으로만 답해줘:\n"
                f"{{\n"
                f'  "order": {labels},\n'
                f'  "logical": "왜 이 순서가 논리적으로 맞는지",\n'
                f'  "visual": "각 프레임에서 어떤 시각적 요소(색상/동작/객체)가 이 순서를 뒷받침하는지",\n'
                f'  "causal": "이벤트 간 인과관계",\n'
                f'  "contrastive": "만약 순서가 다르다면 왜 말이 안 되는지"\n'
                f"}}"
    })

    text = call_gemini_with_retry(contents)
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip()), order
    except:
        return {"order": None, "logical": None, "visual": None,
                "causal": None, "contrastive": None}, order