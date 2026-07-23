"""
Qwen2-VL 최소 동작 테스트.
목적: 모델 로드 + 이미지 하나 넣고 텍스트 생성이 정상적으로 되는지만 확인.
실제 실험 로직(explain_generate.py 연동)은 이게 성공한 뒤에 진행.

사용법: python test_qwen.py
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "./qwen_model"  # huggingface-cli로 받은 로컬 경로. 캐시에서 바로 쓰려면 "Qwen/Qwen2-VL-7B-Instruct"로 변경 가능

print("모델 로드 중... (시간 걸릴 수 있음)")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",  # 여러 GPU에 자동 분산
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("모델 로드 완료")

# 프로젝트 안에 있는 실제 이미지 하나로 테스트 (경로는 환경에 맞게 조정 필요)
import glob
sample_images = glob.glob("data/activitynet_image/*/0.jpg")
if not sample_images:
    raise FileNotFoundError("테스트용 이미지를 못 찾음 — data/activitynet_image 경로 확인 필요")

test_image_path = sample_images[0]
print(f"테스트 이미지: {test_image_path}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_image_path},
            {"type": "text", "text": "이 이미지에 뭐가 보이는지 한 문장으로 설명해줘."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

print("생성 중...")
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n===== 결과 =====")
print(output_text[0])
