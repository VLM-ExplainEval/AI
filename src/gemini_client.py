import os
import re
import time
from dotenv import load_dotenv
from google import genai
from config import MODEL_NAME
from data_loader import load_images_as_base64

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini_order(video_id, shuffled=False, max_retries=5):
    images, indices = load_images_as_base64(video_id, shuffled)
    contents = []
    for img in images:
        contents.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img
            }
        })
    contents.append({
        "text": "These are 8 frames from a video labeled 0 to 7 in the order shown. Arrange them in the correct temporal order they would appear in the original video. Reply with ONLY a Python list like [3,0,1,2,4,5,6,7]. No explanation."
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
                wait = 30 * (attempt + 1)  # 변경: 30, 60, 90, 120초
                print(f"  503 재시도 중... {wait}초 대기 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e

    raise Exception("최대 재시도 횟수 초과")

def parse_order(response_text):
    match = re.search(r'\[[\d,\s]+\]', response_text)
    if match:
        return eval(match.group())
    return None