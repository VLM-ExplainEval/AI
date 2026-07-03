import os
import re
import time
from dotenv import load_dotenv
from google import genai
from config import MODEL_NAME, TEST_JSON, TRAIN_JSON
from data_loader import load_json_files

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_sentences(video_id):
    json_paths = [TEST_JSON, TRAIN_JSON]
    all_data = load_json_files(json_paths)
    for sample in all_data:
        vid = list(sample.keys())[0]
        if vid == video_id:
            return sample[vid]['sentences']
    return None

def ask_gemini_order(video_id, frame_indices, shuffled=False, max_retries=5):
    import random
    sentences = get_sentences(video_id)
    if sentences is None:
        raise Exception(f"{video_id} sentences 없음")

    # frame_indices에 해당하는 sentences만 추출
    selected = [sentences[i] for i in frame_indices]
    n = len(selected)
    labels = [chr(65 + i) for i in range(n)]  # ['A', 'B', 'C']

    # 셔플이면 순서 섞기
    order = list(range(n))
    if shuffled:
        random.shuffle(order)

    # 라벨링: 셔플된 순서로 A,B,C 할당
    labeled = {labels[new_pos]: selected[orig_pos]
               for new_pos, orig_pos in enumerate(order)}

    prompt = f"Q. In which order are the described events shown in the video? " \
             f"Arrange the given list in the order they occur in the video.\n\n"
    for label in labels:
        prompt += f"{label}. {labeled[label]}\n"
    prompt += f"\nReply with ONLY a Python list like {labels}. No explanation."

    contents = [{"text": prompt}]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )
            return response.text, order
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                wait = 45
                print(f"  재시도 중... {wait}초 대기 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e

    raise Exception("최대 재시도 횟수 초과")

def parse_order(response_text):
    if response_text is None:
        return None
    match = re.search(r'\[[^\]]+\]', response_text)
    if match:
        try:
            result = eval(match.group())
            if isinstance(result, list):
                return result
        except:
            return None
    return None