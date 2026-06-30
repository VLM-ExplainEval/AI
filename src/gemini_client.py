import os
import re
import time
from dotenv import load_dotenv
from google import genai
from config import MODEL_NAME
from data_loader import load_images_as_base64

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini_order(video_id, frame_indices, shuffled=False, max_retries=5):
    n = len(frame_indices)
    images, indices = load_images_as_base64(video_id, frame_indices, shuffled)
    contents = []
    for img in images:
        contents.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img
            }
        })
    contents.append({
        "text": f"These are {n} frames from a video labeled 0 to {n-1} in the order shown. Arrange them in the correct temporal order they would appear in the original video. Reply with ONLY a Python list like {list(range(n))}. No explanation."
    })

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )
            return response.text, indices
        except Exception as e:
            if "503" in str(e):
                wait = 60 * (attempt + 1)
                print(f"  503 재시도 중... {wait}초 대기 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e

    raise Exception("최대 재시도 횟수 초과")

def parse_order(response_text):
    if response_text is None:
        return None
    match = re.search(r'\[[\d,\s]+\]', response_text)
    if match:
        return eval(match.group())
    return None