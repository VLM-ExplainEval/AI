"""
실험2 실행 스크립트 (분리된 버전).

기존 experiment2_explain.py를 대체. generate/judge를 독립적으로 켜고 끌 수 있어,
"오늘은 생성만 지켜보고, 채점은 자면서 돌리기" 같은 작업 방식이 가능함.

사용법:
    python run_experiment2.py low --mode generate      # 생성만 하고 CSV(설명) 저장
    python run_experiment2.py low --mode judge --input results/experiment2_gen_low_XXXX.csv
                                                        # 이미 생성된 CSV를 읽어서 채점만
    python run_experiment2.py low --mode both           # 기존과 동일하게 한번에 (기본값)

옵션:
    --model gemini|qwen           생성에 쓸 모델 (기본 gemini, qwen은 추후 구현)
    --judge-model gemini|qwen     채점에 쓸 모델 (기본 gemini)
    --judge-with-images           채점 시 이미지도 같이 보여줄지 여부 (기본: 안 씀, 기존 방식과 동일)
    --n 10                        샘플 수 (기본 10)
"""

import argparse
import csv
import os
import json
from datetime import datetime

from data_loader import load_grouped_data
from gemini_client import get_sentences
from metrics import calc_eta, calc_eta_simple
from config import RESULT_DIR, TEST_JSON, TRAIN_JSON

from explain_generate import generate_org_and_shuf
from explain_judge import judge_explanations, judge_explanations_with_images


def load_all_data():
    return {list(s.keys())[0]: list(s.values())[0]
            for s in json.load(open(TEST_JSON, encoding="utf-8")) + json.load(open(TRAIN_JSON, encoding="utf-8"))}


def run_generate(group, n_samples, model="gemini"):
    json_paths = [TEST_JSON, TRAIN_JSON]
    samples = load_grouped_data(json_paths, group=group, n=n_samples)
    print(f"[생성] {group} 그룹 {len(samples)}개, model={model}")

    csv_rows = []
    for i, (video_id, frame_indices) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {video_id} | 프레임: {frame_indices}")
        sentences = get_sentences(video_id)
        selected = [sentences[j] for j in frame_indices]

        org_result, shuf_result = generate_org_and_shuf(video_id, frame_indices, model=model)

        print(f"  Org EM: {org_result['em']}, 예측: {org_result['pred']}")
        print(f"  Shuf EM: {shuf_result['em']}, 예측: {shuf_result['pred']}")

        csv_rows.append({
            "video_id": video_id,
            "frame_indices": str(frame_indices),
            "event_A": selected[0],
            "event_B": selected[1],
            "event_C": selected[2],
            "org_em": org_result["em"],
            "shuf_em": shuf_result["em"],
            "org_pred": str(org_result["pred"]),
            "shuf_pred": str(shuf_result["pred"]),
            "logical_explanation": org_result["explanations"]["logical"],
            "visual_explanation": org_result["explanations"]["visual"],
            "causal_explanation": org_result["explanations"]["causal"],
            "contrastive_explanation": org_result["explanations"]["contrastive"],
            "shuf_logical_explanation": shuf_result["explanations"]["logical"],
            "shuf_visual_explanation": shuf_result["explanations"]["visual"],
            "shuf_causal_explanation": shuf_result["explanations"]["causal"],
            "shuf_contrastive_explanation": shuf_result["explanations"]["contrastive"],
        })

    timestamp = datetime.now().strftime("%m%d_%H%M")
    csv_path = os.path.join(RESULT_DIR, f"experiment2_gen_{group}_{model}_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[생성 완료] 저장: {csv_path}")

    # 실험1과 동일하게, 생성이 끝난 시점에 EM 기반 η를 바로 계산해서 출력
    # (채점 점수는 필요 없이 org_em/shuf_em/org_pred/shuf_pred만으로 계산 가능)
    rows_for_eta = [{
        "org_em": r["org_em"], "shuf_em": r["shuf_em"],
        "org_pred": r["org_pred"], "shuf_pred": r["shuf_pred"],
    } for r in csv_rows]

    org_acc = sum(int(r["org_em"]) for r in csv_rows) / len(csv_rows) * 100
    shuf_acc = sum(int(r["shuf_em"]) for r in csv_rows) / len(csv_rows) * 100
    eta_vector = calc_eta(rows_for_eta)
    eta_simple = calc_eta_simple(org_acc, shuf_acc)

    print(f"\n===== {group} 그룹 실험2 생성 결과 (model={model}) =====")
    print(f"샘플 수: {len(csv_rows)}")
    print(f"Org EM: {org_acc:.2f}%")
    print(f"Shuf EM: {shuf_acc:.2f}%")
    print(f"η (VECTOR 공식): {eta_vector:.2f}%" if eta_vector is not None else "η (VECTOR 공식): 계산 불가")
    print(f"η (단순 공식): {eta_simple:.2f}%" if eta_simple is not None else "η (단순 공식): 계산 불가")

    return csv_path, csv_rows


def _judge_one(explanations, frame_indices, video_id, all_data, judge_model, with_images, shuffled):
    if with_images:
        return judge_explanations_with_images(
            explanations, frame_indices, video_id, all_data,
            shuffled=shuffled, judge_model=judge_model
        )
    else:
        return judge_explanations(
            explanations, frame_indices, video_id, all_data, judge_model=judge_model
        )


def run_judge(csv_path=None, csv_rows=None, judge_model="gemini", with_images=False,
              group=None, judge_shuf=True):
    """
    judge_shuf=True(기본값): Org 설명뿐 아니라 Shuf 설명도 채점한다.
                            Shuf 설명 점수는 shuf_logical_score 등 shuf_ 접두어로 저장됨.
    judge_shuf=False: 기존과 동일하게 Org 설명만 채점한다.
    """
    if csv_rows is None:
        if csv_path is None:
            raise ValueError("csv_path 또는 csv_rows 중 하나는 필요합니다.")
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

    all_data = load_all_data()
    print(f"[채점] {len(csv_rows)}개 샘플, judge_model={judge_model}, "
          f"with_images={with_images}, judge_shuf={judge_shuf}")

    for i, row in enumerate(csv_rows):
        video_id = row["video_id"]
        frame_indices = eval(row["frame_indices"]) if isinstance(row["frame_indices"], str) else row["frame_indices"]

        org_explanations = {
            "logical": row.get("logical_explanation"),
            "visual": row.get("visual_explanation"),
            "causal": row.get("causal_explanation"),
            "contrastive": row.get("contrastive_explanation"),
        }

        print(f"[{i+1}/{len(csv_rows)}] {video_id} 채점 중 (Org)...")
        org_scores = _judge_one(org_explanations, frame_indices, video_id, all_data,
                                 judge_model, with_images, shuffled=False)
        print(f"  Org 점수: {org_scores}")

        row["logical_score"] = org_scores.get("logical_score")
        row["visual_score"] = org_scores.get("visual_score")
        row["causal_score"] = org_scores.get("causal_score")
        row["contrastive_score"] = org_scores.get("contrastive_score")

        if judge_shuf:
            shuf_explanations = {
                "logical": row.get("shuf_logical_explanation"),
                "visual": row.get("shuf_visual_explanation"),
                "causal": row.get("shuf_causal_explanation"),
                "contrastive": row.get("shuf_contrastive_explanation"),
            }
            print(f"[{i+1}/{len(csv_rows)}] {video_id} 채점 중 (Shuf)...")
            shuf_scores = _judge_one(shuf_explanations, frame_indices, video_id, all_data,
                                      judge_model, with_images, shuffled=True)
            print(f"  Shuf 점수: {shuf_scores}")

            row["shuf_logical_score"] = shuf_scores.get("logical_score")
            row["shuf_visual_score"] = shuf_scores.get("visual_score")
            row["shuf_causal_score"] = shuf_scores.get("causal_score")
            row["shuf_contrastive_score"] = shuf_scores.get("contrastive_score")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    suffix = "img" if with_images else "text"
    group_str = group or "unknown"
    out_path = os.path.join(RESULT_DIR, f"experiment2_judge_{group_str}_{judge_model}_{suffix}_{timestamp}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[채점 완료] 저장: {out_path}")

    print_summary(csv_rows, judge_shuf=judge_shuf)
    return out_path, csv_rows


def print_summary(csv_rows, judge_shuf=True):
    rows_for_eta = [{
        "org_em": r["org_em"], "shuf_em": r["shuf_em"],
        "org_pred": r["org_pred"], "shuf_pred": r["shuf_pred"],
    } for r in csv_rows]

    org_acc = sum(int(r["org_em"]) for r in csv_rows) / len(csv_rows) * 100
    shuf_acc = sum(int(r["shuf_em"]) for r in csv_rows) / len(csv_rows) * 100
    eta_simple = calc_eta_simple(org_acc, shuf_acc)
    eta_vector = calc_eta(rows_for_eta)

    print("\n===== 결과 요약 =====")
    print(f"샘플 수: {len(csv_rows)}")
    print(f"Org EM: {org_acc:.2f}%")
    print(f"Shuf EM: {shuf_acc:.2f}%")
    print(f"η (단순): {eta_simple:.2f}%" if eta_simple is not None else "η (단순): 계산불가")
    print(f"η (VECTOR): {eta_vector:.2f}%" if eta_vector is not None else "η (VECTOR): 계산불가")

    print("\n[Org 설명 품질]")
    for key in ["logical_score", "visual_score", "causal_score", "contrastive_score"]:
        valid = [r[key] for r in csv_rows if r.get(key) is not None]
        avg = sum(float(v) for v in valid) / len(valid) if valid else 0
        print(f"{key}: 평균 {avg:.2f}점 ({len(valid)}/{len(csv_rows)}개)")

    if judge_shuf:
        print("\n[Shuf 설명 품질]")
        for key in ["shuf_logical_score", "shuf_visual_score", "shuf_causal_score", "shuf_contrastive_score"]:
            valid = [r[key] for r in csv_rows if r.get(key) is not None]
            avg = sum(float(v) for v in valid) / len(valid) if valid else 0
            print(f"{key}: 평균 {avg:.2f}점 ({len(valid)}/{len(csv_rows)}개)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", nargs="?", default="low", choices=["low", "high"])
    parser.add_argument("--mode", default="both", choices=["generate", "judge", "both"])
    parser.add_argument("--model", default="gemini")
    parser.add_argument("--judge-model", default="gemini")
    parser.add_argument("--judge-with-images", action="store_true")
    parser.add_argument("--no-judge-shuf", action="store_true",
                         help="지정하면 Shuf 설명은 채점하지 않고 Org만 채점 (기존 방식)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--input", default=None, help="mode=judge일 때 읽을 CSV 경로")
    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args.group, args.n, model=args.model)

    elif args.mode == "judge":
        run_judge(csv_path=args.input, judge_model=args.judge_model,
                   with_images=args.judge_with_images, group=args.group,
                   judge_shuf=not args.no_judge_shuf)

    elif args.mode == "both":
        csv_path, csv_rows = run_generate(args.group, args.n, model=args.model)
        run_judge(csv_rows=csv_rows, judge_model=args.judge_model,
                   with_images=args.judge_with_images, group=args.group,
                   judge_shuf=not args.no_judge_shuf)


if __name__ == "__main__":
    main()