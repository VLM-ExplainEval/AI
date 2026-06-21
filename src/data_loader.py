import os
import json
import base64
import random
from config import IMAGE_DIR, TEST_JSON, NUM_FRAMES

# test.json 파일을 불러오는 함수
def load_test_data():
    with open(TEST_JSON, "r") as f:
        return json.load(f)

# shuffled=True이면 프레임 순서 랜덤으로 섞음
# 영상 폴더에서 8개 프레임 이미지를 읽어서 base64로 변환
def load_images_as_base64(video_id, shuffled=False):
    folder = os.path.join(IMAGE_DIR, video_id)
    indices = list(range(NUM_FRAMES))
    if shuffled:
        random.shuffle(indices)
    images = []
    for i in indices:
        path = os.path.join(folder, f"{i}.jpg")
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images.append(encoded)
    return images, indices