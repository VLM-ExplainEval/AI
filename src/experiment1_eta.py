import os
import json
import base64
from dotenv import load_dotenv
from google import genai
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "activitynet_image")

# 데이터 로드
with open(os.path.join(DATA_DIR, "test.json"), "r") as f:
    data = json.load(f)

def load_images_as_base64(video_id, shuffled=False):
    folder = os.path.join(IMAGE_DIR, video_id)
    indices = list(range(8))
    if shuffled:
        import random
        random.shuffle(indices)
    
    images = []
    for i in indices:
        path = os.path.join(folder, f"{i}.jpg")
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images.append(encoded)
    return images, indices

def ask_gemini_order(video_id, shuffled=False):
    images, indices = load_images_as_base64(video_id, shuffled)
    
    contents = []
    for i, img in enumerate(images):
        contents.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img
            }
        })
    
    contents.append({
        "text": "These are 8 frames from a video labeled 0 to 7 in the order shown. Arrange them in the correct temporal order they would appear in the original video. Reply with ONLY a Python list like [3,0,1,2,4,5,6,7]. No explanation."
    })
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    return response.text, indices

def parse_order(response_text):
    match = re.search(r'\[[\d,\s]+\]', response_text)
    if match:
        return eval(match.group())
    return None

# 테스트
sample = data[0]
video_id = list(sample.keys())[0]
info = sample[video_id]  # 이거 추가

print(f"영상 ID: {video_id}")
print("Org 테스트 중...")
response, _ = ask_gemini_order(video_id, shuffled=False)
print(f"Gemini 응답: {response}")
parsed = parse_order(response)
print(f"파싱된 순서: {parsed}")
print("=======================================================")

gt = list(range(len(info['sentences'])))
print(f"정답: {gt}")
print(f"Gemini 예측: {parsed}")
print("=======================================================")
print(f"정확히 맞췄나? {gt == parsed}")