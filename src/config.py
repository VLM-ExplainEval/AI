# 경로랑 모델명 같은 설정값 모아두는 파일
import os

DATA_DIR = "/home/undergraduate/20231372_TY/VLM-ExplainEval/data"
IMAGE_DIR = "/home/undergraduate/20231372_TY/VLM-ExplainEval/data/activitynet_image"
TEST_JSON = os.path.join(DATA_DIR, "test.json")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TRAIN_IMAGE_DIR = "/home/undergraduate/20231372_TY/VLM-ExplainEval/data/8frames_train"
RESULT_DIR = "/home/undergraduate/20221373_YY/VLM-ExplainEval/results"
MODEL_NAME = "gemini-2.5-flash"