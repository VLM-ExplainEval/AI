"""
사람 검증 스크립트 — Gemini가 채점한 결과와 사람이 직접 매긴 점수를 비교합니다.

사용법:
    python human_validation.py <csv파일경로> [--n 15] [--seed 42]

예시:
    python human_validation.py experiment2_judge_low_gemini_text_0709_0810.csv --n 15

동작:
    1. CSV에서 무작위로 n개 샘플을 뽑음 (seed 고정, 재현 가능)
    2. 각 샘플의 4가지 설명(logical/visual/causal/contrastive)을 한 항목씩 터미널에 보여줌
    3. 사람이 1/3/5점 중 하나를 입력
    4. 전부 끝나면 Gemini 점수와 사람 점수의 일치도(정확 일치율, 허용오차 내 일치율, 상관계수)를 출력
    5. 결과를 CSV로 저장 (human_validation_result_YYYYMMDD_HHMM.csv)

중간에 중단해도(Ctrl+C) 그때까지 입력한 결과는 저장됩니다.
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime

ITEM_LABELS = {
    "logical": "① 논리적 근거",
    "visual": "② 시각적 그라운딩",
    "causal": "③ 인과적 설명",
    "contrastive": "④ 대조적 설명",
}

VALID_SCORES = {"1", "3", "5"}


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def pick_samples(rows, n, seed):
    random.seed(seed)
    if len(rows) <= n:
        return rows
    return random.sample(rows, n)


def ask_score(prompt_label):
    while True:
        val = input(f"    → {prompt_label} 점수 입력 (1 / 3 / 5, s=건너뛰기, q=종료): ").strip().lower()
        if val == "q":
            return "QUIT"
        if val == "s":
            return None
        if val in VALID_SCORES:
            return int(val)
        print("      ⚠ 1, 3, 5 중 하나만 입력해주세요. (또는 s=건너뛰기, q=종료)")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    parser = argparse.ArgumentParser(description="LLM 채점 결과 사람 검증 스크립트")
    parser.add_argument("csv_path", help="채점 결과 CSV 파일 경로")
    parser.add_argument("--n", type=int, default=15, help="검증할 샘플 수 (기본 15개)")
    parser.add_argument("--seed", type=int, default=42, help="샘플 추출 seed (기본 42)")
    parser.add_argument("--shuf", action="store_true", help="Org 대신 Shuf(shuf_*) 컬럼을 검증 대상으로 함")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"파일을 찾을 수 없습니다: {args.csv_path}")
        sys.exit(1)

    rows = load_rows(args.csv_path)
    samples = pick_samples(rows, args.n, args.seed)

    prefix = "shuf_" if args.shuf else ""
    condition_label = "Shuf" if args.shuf else "Org"

    print(f"\n총 {len(samples)}개 샘플을 검증합니다. (조건: {condition_label}, seed={args.seed})")
    print("각 항목마다 설명 텍스트를 보여드리고, 1/3/5점 중 하나를 입력받습니다.")
    print("중간에 그만두려면 'q'를 입력하세요. 특정 항목만 건너뛰려면 's'를 입력하세요.\n")
    input("준비되면 Enter를 누르세요...")

    results = []
    quit_early = False

    for i, row in enumerate(samples):
        if quit_early:
            break

        clear_screen()
        video_id = row.get("video_id", "unknown")
        print(f"========== 샘플 {i+1}/{len(samples)} | video_id: {video_id} ==========\n")

        # 이벤트 원문 캡션도 같이 보여줌 (판단에 참고)
        for ev in ["event_A", "event_B", "event_C"]:
            if ev in row:
                print(f"  {ev}: {row[ev]}")
        print()

        row_result = {
            "video_id": video_id,
            "condition": condition_label,
        }

        for item_key, item_label in ITEM_LABELS.items():
            col_name = f"{prefix}{item_key}_explanation"
            score_col_name = f"{prefix}{item_key}_score"

            explanation = row.get(col_name, "(텍스트 없음)")
            gemini_score = row.get(score_col_name, "N/A")

            print(f"--- {item_label} ---")
            print(explanation)
            print()

            human_score = ask_score(item_label)
            if human_score == "QUIT":
                quit_early = True
                break

            row_result[f"{item_key}_gemini_score"] = gemini_score
            row_result[f"{item_key}_human_score"] = human_score
            print()

        if not quit_early:
            results.append(row_result)

    if not results:
        print("\n입력된 결과가 없어 종료합니다.")
        return

    # ---- 결과 저장 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"human_validation_result_{timestamp}.csv"
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ---- 일치도 계산 ----
    clear_screen()
    print(f"===== 검증 완료: {len(results)}개 샘플 처리됨 =====\n")
    print(f"결과 저장: {out_path}\n")

    for item_key, item_label in ITEM_LABELS.items():
        pairs = []
        for r in results:
            g = r.get(f"{item_key}_gemini_score")
            h = r.get(f"{item_key}_human_score")
            if g is None or h is None or g == "N/A":
                continue
            try:
                g_val = float(g)
                h_val = float(h)
            except ValueError:
                continue
            pairs.append((g_val, h_val))

        if not pairs:
            print(f"{item_label}: 비교 가능한 데이터 없음\n")
            continue

        exact_match = sum(1 for g, h in pairs if g == h) / len(pairs) * 100
        within_2 = sum(1 for g, h in pairs if abs(g - h) <= 2) / len(pairs) * 100
        avg_diff = sum(abs(g - h) for g, h in pairs) / len(pairs)

        print(f"{item_label} (n={len(pairs)})")
        print(f"  정확 일치율     : {exact_match:.1f}%")
        print(f"  ±2점 이내 일치율: {within_2:.1f}%")
        print(f"  평균 절대 오차   : {avg_diff:.2f}점")
        print()

    print("이 결과를 보고서의 'LLM Judge 신뢰도 검증' 섹션에 반영하시면 됩니다.")


if __name__ == "__main__":
    main()
