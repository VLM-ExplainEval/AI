# 경로랑 모델명 같은 설정값 모아두는 파일
import os

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "activitynet_image")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "8frames_train")
RESULT_DIR = "results"
MODEL_NAME = "gemini-2.5-flash"
