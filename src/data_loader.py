import os
import json
import base64
import random
from config import IMAGE_DIR, TRAIN_IMAGE_DIR, TEST_JSON

def load_test_data():
    with open(TEST_JSON, "r") as f:
        return json.load(f)

def load_json_files(paths):
    all_data = []
    for path in paths:
        with open(path, "r") as f:
            all_data.extend(json.load(f))
    return all_data

def get_image_folder(video_id):
    test_path = os.path.join(IMAGE_DIR, video_id)
    train_path = os.path.join(TRAIN_IMAGE_DIR, video_id)
    if os.path.exists(test_path):
        return test_path
    elif os.path.exists(train_path):
        return train_path
    else:
        return None

def get_filename(folder, idx):
    """폴더 내 파일명 형식 자동 판별 (0.jpg vs 00.jpg)"""
    if os.path.exists(os.path.join(folder, f"{idx}.jpg")):
        return f"{idx}.jpg"
    elif os.path.exists(os.path.join(folder, f"{idx:02d}.jpg")):
        return f"{idx:02d}.jpg"
    else:
        return None

def load_grouped_data(json_paths, group='low', n=None):
    all_data = load_json_files(json_paths)
    result = []

    for sample in all_data:
        video_id = list(sample.keys())[0]
        info = sample[video_id]
        rel = info['relation'][0]
        count_one = rel.count('1')

        folder = get_image_folder(video_id)
        if folder is None:
            continue

        if group == 'low' and count_one == 1:
            one_idx = rel.index('1')
            num_events = len(info['sentences'])
            candidates = [i for i in range(num_events) if i < one_idx or i > one_idx + 1]
            if len(candidates) >= 3:
                frames = candidates[:3]
                if all(get_filename(folder, f) is not None for f in frames):
                    result.append((video_id, frames))

        elif group == 'high' and count_one >= 3:
            causal_indices = [i for i, c in enumerate(rel) if c == '1']
            num_events = len(info['sentences'])  # 실제 이벤트 수
            last_idx = num_events - 1            # 마지막 이벤트 인덱스
            frames = causal_indices[:2] + [last_idx]
            if len(frames) == 3:
                folder = get_image_folder(video_id)
                if all(get_filename(folder, f) is not None for f in frames):
                    result.append((video_id, frames))

        if n is not None and len(result) >= n:
            break

    return result

def load_images_as_base64(video_id, frame_indices, shuffled=False):
    folder = get_image_folder(video_id)
    if folder is None:
        raise FileNotFoundError(f"{video_id} 이미지 폴더를 찾을 수 없음")

    order = list(range(len(frame_indices)))
    if shuffled:
        random.shuffle(order)

    images = []
    for rel_pos in order:
        actual_frame = frame_indices[rel_pos]
        filename = get_filename(folder, actual_frame)
        if filename is None:
            raise FileNotFoundError(f"{video_id}/{actual_frame}.jpg 없음")
        path = os.path.join(folder, filename)
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images.append(encoded)

    return images, order