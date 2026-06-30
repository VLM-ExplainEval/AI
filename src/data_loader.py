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
    """video_id가 test/train 어느 폴더에 있는지 자동 판별"""
    test_path = os.path.join(IMAGE_DIR, video_id)
    train_path = os.path.join(TRAIN_IMAGE_DIR, video_id)
    if os.path.exists(test_path):
        return test_path
    elif os.path.exists(train_path):
        return train_path
    else:
        return None

def load_grouped_data(json_paths, group='low', n=10):
    """
    group: 'low' (relation 1이 정확히 1개) or 'high' (연속 1이 3개 이상)
    n: 추출할 샘플 수
    반환: [(video_id, [frame_idx1, frame_idx2, frame_idx3]), ...]
    """
    all_data = load_json_files(json_paths)
    result = []

    for sample in all_data:
        video_id = list(sample.keys())[0]
        info = sample[video_id]
        rel = info['relation'][0]
        count_one = rel.count('1')

        if get_image_folder(video_id) is None:
            continue

        max_c = cur = 0
        cur_start = -1
        start_idx = -1
        for idx, c in enumerate(rel):
            if c == '1':
                if cur == 0:
                    cur_start = idx
                cur += 1
                if cur > max_c:
                    max_c = cur
                    start_idx = cur_start
            else:
                cur = 0

        if group == 'low' and count_one == 1:
            one_idx = rel.index('1')
            candidates = [i for i in range(8) if i < one_idx or i > one_idx + 1]
            if len(candidates) >= 3:
                result.append((video_id, candidates[:3]))

        elif group == 'high' and max_c >= 3:
            result.append((video_id, [start_idx, start_idx + 1, start_idx + 2]))

        if len(result) >= n:
            break

    return result

def load_images_as_base64(video_id, frame_indices, shuffled=False):
    folder = get_image_folder(video_id)
    if folder is None:
        raise FileNotFoundError(f"{video_id} 이미지 폴더를 찾을 수 없음")
    
    # frame_indices = 실제 프레임 번호 (예: [0,3,4])
    order = list(range(len(frame_indices)))  # [0, 1, 2] 상대 위치
    if shuffled:
        random.shuffle(order)
    
    images = []
    for rel_pos in order:
        actual_frame = frame_indices[rel_pos]
        path = os.path.join(folder, f"{actual_frame}.jpg")
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images.append(encoded)
    
    return images, order  # order는 0~2 범위의 상대 인덱스