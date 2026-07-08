import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "activitynet_image")
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "8frames_train")
RESULT_DIR = os.path.join(BASE_DIR, "results")

TEST_JSON = os.path.join(DATA_DIR, "test.json")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")

MODEL_NAME = "gemini-2.5-flash"

os.makedirs(RESULT_DIR, exist_ok=True)