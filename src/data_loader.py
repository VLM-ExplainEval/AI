import os
import json
import base64
import random
from config import IMAGE_DIR, TEST_JSON

# test.json 파일을 불러오는 함수
def load_test_data():
    with open(TEST_JSON, "r") as f:
        return json.load(f)

# train.json, test.json 등 여러 파일 합쳐서 불러오는 함수
def load_json_files(paths):
    all_data = []
    for path in paths:
        with open(path, "r") as f:
            all_data.extend(json.load(f))
    return all_data

# relation 필드 기준으로 저인과/고인과 그룹 + 3프레임 추출
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

        # 최대 연속 1 구간 찾기
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
            # relation은 길이 7 (8개 이벤트), 1이 있는 위치(one_idx, one_idx+1) 제외하고 0인 구간에서 3프레임
            candidates = [i for i in range(8) if i < one_idx or i > one_idx + 1]
            if len(candidates) >= 3:
                frames = candidates[:3]
                result.append((video_id, frames))

        elif group == 'high' and max_c >= 3:
            # 연속 111 구간의 첫 위치부터 3프레임 (start_idx, start_idx+1, start_idx+2)
            frames = [start_idx, start_idx + 1, start_idx + 2]
            result.append((video_id, frames))

        if len(result) >= n:
            break

    return result

# frame_indices로 지정된 프레임만 읽어서 base64로 변환
# shuffled=True면 그 3개 프레임의 순서만 섞음
def load_images_as_base64(video_id, frame_indices, shuffled=False):
    folder = os.path.join(IMAGE_DIR, video_id)
    indices = list(frame_indices)
    if shuffled:
        random.shuffle(indices)
    images = []
    for i in indices:
        path = os.path.join(folder, f"{i}.jpg")
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images.append(encoded)
    return images, indices