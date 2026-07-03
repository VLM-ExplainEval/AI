import csv
import os
import sys
import time
import json
from data_loader import load_grouped_data
from gemini_client import ask_gemini_order_with_explanation, get_sentences, call_gemini_with_retry
from metrics import get_gt_from_order, exact_match, calc_eta, calc_eta_simple
from config import RESULT_DIR, TEST_JSON, TRAIN_JSON
from datetime import datetime

GROUP = sys.argv[1] if len(sys.argv) > 1 else "low"
N_SAMPLES = 10

json_paths = [TEST_JSON, TRAIN_JSON]
samples = load_grouped_data(json_paths, group=GROUP, n=N_SAMPLES)

print(f"{GROUP} 그룹 {len(samples)}개 실험 2 시작")

all_data = {list(s.keys())[0]: list(s.values())[0]
            for s in json.load(open(TEST_JSON)) + json.load(open(TRAIN_JSON))}


def judge_explanations(explanations, frame_indices, video_id, all_data):
    info = all_data.get(video_id, {})
    existence = info.get('Existence', [])
    cot = info.get('cot', [])

    existence_texts = []
    cot_texts = []
    labels = ['A', 'B', 'C']
    for j, idx in enumerate(frame_indices):
        if idx < len(existence):
            existence_texts.append(f"이벤트 {labels[j]}: {existence[idx]}")
        if idx < len(cot):
            cot_texts.append(f"이벤트 {labels[j]}: {cot[idx]}")

    prompt = f"""다음 설명들을 엄격하게 평가해줘.

정답 시각적 요소 (Existence):
{chr(10).join(existence_texts)}

정답 인과 설명 (CoT):
{chr(10).join(cot_texts)}

모델이 생성한 설명:
- 논리적 근거: {explanations.get('logical', '')}
- 시각적 그라운딩: {explanations.get('visual', '')}
- 인과적 설명: {explanations.get('causal', '')}
- 대조적 설명: {explanations.get('contrastive', '')}

채점 기준 (엄격하게):
- logical_score: 논리적 흐름이 명확한가?
  1=근거없음, 2=매우일반적, 3=보통, 4=명확, 5=매우구체적
- visual_score: Existence에 나열된 구체적 요소(색상/객체/위치)를 실제로 언급했는가?
  1=Existence 요소 전혀 언급안함
  2=배경(실내/실외)만 언급
  3=일반적 객체만 언급(사람, 물건 등)
  4=Existence 요소 일부 구체적 언급
  5=Existence 요소 대부분 구체적으로 언급
- causal_score: CoT의 인과관계를 실제로 설명했는가?
  1=인과관계 없음, 3=일반적 인과, 5=CoT와 유사한 구체적 인과
- contrastive_score: 다른 순서가 왜 틀렸는지 구체적으로 반박했는가?
  1=반박없음, 3=일반적반박, 5=구체적이고납득가능한반박

JSON으로만 답해줘:
{{"logical_score": 점수, "visual_score": 점수, "causal_score": 점수, "contrastive_score": 점수}}"""

    try:
        text = call_gemini_with_retry([{"text": prompt}])
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except:
        return {"logical_score": None, "visual_score": None,
                "causal_score": None, "contrastive_score": None}


rows = []           # calc_eta 계산용 (org_pred, shuf_pred가 리스트 그대로 들어있음)
csv_rows = []        # CSV 저장용 (org_pred, shuf_pred가 문자열로 변환됨)

for i, (video_id, frame_indices) in enumerate(samples):
    print(f"[{i+1}/{len(samples)}] {video_id} | 프레임: {frame_indices}")

    sentences = get_sentences(video_id)
    selected = [sentences[j] for j in frame_indices]

    # Org
    try:
        result_org, order = ask_gemini_order_with_explanation(video_id, frame_indices, shuffled=False)
        org_pred = result_org.get("order")
        gt = get_gt_from_order(order)
        org_em = exact_match(gt, org_pred)
        print(f"  Org EM: {org_em}, 예측: {org_pred}")
    except Exception as e:
        print(f"  Org 에러: {e}")
        result_org = {"order": None, "logical": None, "visual": None, "causal": None, "contrastive": None}
        org_pred = None
        org_em = 0

    # Shuf
    try:
        result_shuf, shuf_order = ask_gemini_order_with_explanation(video_id, frame_indices, shuffled=True)
        shuf_pred = result_shuf.get("order")
        shuf_gt = get_gt_from_order(shuf_order)
        shuf_em = exact_match(shuf_gt, shuf_pred)
        print(f"  Shuf EM: {shuf_em}, 예측: {shuf_pred}")
    except Exception as e:
        print(f"  Shuf 에러: {e}")
        result_shuf = {"order": None, "logical": None, "visual": None, "causal": None, "contrastive": None}
        shuf_pred = None
        shuf_em = 0

    time.sleep(5)

    # Judge 채점 (Org 설명 기준)
    try:
        scores = judge_explanations(result_org, frame_indices, video_id, all_data)
        print(f"  점수: {scores}")
    except Exception as e:
        print(f"  Judge 에러: {e}")
        scores = {"logical_score": None, "visual_score": None,
                  "causal_score": None, "contrastive_score": None}

    # calc_eta용 - 리스트 그대로 저장
    rows.append({
        "video_id": video_id,
        "org_em": org_em,
        "shuf_em": shuf_em,
        "org_pred": org_pred,
        "shuf_pred": shuf_pred,
    })

    # CSV 저장용 - 문자열로 변환
    csv_rows.append({
        "video_id": video_id,
        "frame_indices": str(frame_indices),
        "event_A": selected[0],
        "event_B": selected[1],
        "event_C": selected[2],
        "org_em": org_em,
        "shuf_em": shuf_em,
        "org_pred": str(org_pred),
        "shuf_pred": str(shuf_pred),
        "logical_explanation": result_org.get("logical"),
        "visual_explanation": result_org.get("visual"),
        "causal_explanation": result_org.get("causal"),
        "contrastive_explanation": result_org.get("contrastive"),
        "logical_score": scores.get("logical_score"),
        "visual_score": scores.get("visual_score"),
        "causal_score": scores.get("causal_score"),
        "contrastive_score": scores.get("contrastive_score"),
    })

    time.sleep(10)

# 결과 계산
org_acc = sum(int(r["org_em"]) for r in rows) / len(rows) * 100
shuf_acc = sum(int(r["shuf_em"]) for r in rows) / len(rows) * 100
eta_simple = calc_eta_simple(org_acc, shuf_acc)
eta_vector = calc_eta(rows)

print(f"\n===== {GROUP} 그룹 실험 2 결과 =====")
print(f"샘플 수: {len(rows)}")
print(f"Org EM: {org_acc:.2f}%")
print(f"Shuf EM: {shuf_acc:.2f}%")
print(f"η (단순): {eta_simple:.2f}%" if eta_simple is not None else "η (단순): 계산불가")
print(f"η (VECTOR): {eta_vector:.2f}%" if eta_vector is not None else "η (VECTOR): 계산불가")
print()
print("===== 설명 품질 점수 =====")
for key in ["logical_score", "visual_score", "causal_score", "contrastive_score"]:
    valid = [r[key] for r in csv_rows if r[key] is not None]
    avg = sum(valid) / len(valid) if valid else 0
    print(f"{key}: 평균 {avg:.2f}점 ({len(valid)}/{len(csv_rows)}개)")

# CSV 저장
timestamp = datetime.now().strftime("%m%d_%H%M")
csv_path = os.path.join(RESULT_DIR, f"experiment2_{GROUP}_{timestamp}.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\n저장 완료: {csv_path}")