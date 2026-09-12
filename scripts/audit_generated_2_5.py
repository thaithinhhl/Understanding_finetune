#!/usr/bin/env python3
"""Kiểm tra data task 2.5 vừa sinh có bị overfit/rò rỉ vào benchmark không.

Ba phép kiểm:

1. RÒ RỈ BENCHMARK — đo trùng lặp n-gram và Jaccard giữa câu hỏi sinh ra và câu hỏi
   trong benchmark. Data sinh độc lập thì phải gần như không trùng.

2. BASELINE BAG-OF-WORDS — huấn luyện một bộ phân loại đa nhãn thô sơ chỉ dựa trên
   từ khoá (Naive Bayes trên unigram), đánh giá chéo ngay trên data sinh ra. Nếu nó
   đạt F1 cao nghĩa là data vẫn "giải được bằng tra từ điển", model sẽ học mẹo thay
   vì học ngữ nghĩa — phải sinh lại. (Chính phép thử này từng phát hiện bộ v2 đạt
   F1 86.6%.)

3. CẤU TRÚC — kiểm tra lại bệnh của bộ v3: tỷ lệ dùng từ nối liệt kê và tương quan
   giữa độ dài câu với số nhãn.
"""

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BANNED_CONNECTORS = ["đồng thời", "ngoài ra", "bên cạnh đó", "thêm nữa", "luôn tiện",
                     "còn nữa", "với lại", "tiện thể", "nhân tiện"]
INTENTS = ["chitchat", "comparative_analysis", "document_relationship", "document_retrieval",
           "external_analysis", "general", "legal_query", "stats_summary"]


def strip_accents(t: str) -> str:
    t = t.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def toks(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def ngrams(t: str, n: int) -> set[tuple]:
    w = toks(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_generated(path: Path) -> list[tuple[str, list[str]]]:
    out = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        q = r["messages"][1]["content"].split("Current turn:")[-1].strip()
        out.append((q, json.loads(r["messages"][2]["content"])["intents"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path, default=Path("vlegal_task_2_5_test.jsonl"))
    args = ap.parse_args()

    gen = load_generated(args.generated)
    print(f"Data sinh ra: {len(gen)} mẫu\n")

    # ---------- 1. Rò rỉ benchmark ----------
    print("=" * 70)
    print("1. KIEM TRA RO RI BENCHMARK")
    if args.benchmark.exists():
        bench = [json.loads(l)["question"] for l in args.benchmark.open(encoding="utf-8") if l.strip()]
        bench_5g = set()
        for q in bench:
            bench_5g |= ngrams(q, 5)
        bench_sets = [set(toks(q)) for q in bench]

        shared_5g = sum(1 for q, _ in gen if ngrams(q, 5) & bench_5g)
        max_j = []
        for q, _ in gen:
            s = set(toks(q))
            best = max((len(s & b) / len(s | b) if (s | b) else 0) for b in bench_sets)
            max_j.append(best)
        max_j.sort()
        n = len(max_j)
        print(f"   Cau chia se >=1 cum 5-gram voi benchmark : {shared_5g}/{n} = {shared_5g/n:.1%}")
        print(f"   Jaccard cao nhat vs benchmark: median={max_j[n//2]:.2f} "
              f"p95={max_j[int(n*0.95)]:.2f} max={max_j[-1]:.2f}")
        print(f"   So cau Jaccard >= 0.7: {sum(1 for j in max_j if j >= 0.7)}")
        print("   -> DAT" if max_j[int(n * 0.95)] < 0.5 else "   -> CAN XEM LAI")
    else:
        print(f"   (bo qua: khong tim thay {args.benchmark})")

    # ---------- 2. Baseline bag-of-words ----------
    print("\n" + "=" * 70)
    print("2. BASELINE BAG-OF-WORDS (Naive Bayes unigram, 5-fold)")
    folds = 5
    tp = fp = fn = 0
    for f in range(folds):
        train = [x for i, x in enumerate(gen) if i % folds != f]
        test = [x for i, x in enumerate(gen) if i % folds == f]
        # đếm tần suất từ theo từng nhãn (one-vs-rest)
        pos_cnt = {i: Counter() for i in INTENTS}
        neg_cnt = {i: Counter() for i in INTENTS}
        pos_n = {i: 0 for i in INTENTS}
        for q, labs in train:
            w = Counter(toks(q))
            for i in INTENTS:
                if i in labs:
                    pos_cnt[i] += w
                    pos_n[i] += 1
                else:
                    neg_cnt[i] += w
        n_train = len(train)
        for q, gold in test:
            w = toks(q)
            pred = set()
            for i in INTENTS:
                if pos_n[i] == 0 or pos_n[i] == n_train:
                    continue
                prior = pos_n[i] / n_train
                pt, nt = sum(pos_cnt[i].values()), sum(neg_cnt[i].values())
                V = len(set(pos_cnt[i]) | set(neg_cnt[i])) + 1
                lp, ln = math.log(prior), math.log(1 - prior)
                for t in w:
                    lp += math.log((pos_cnt[i][t] + 1) / (pt + V))
                    ln += math.log((neg_cnt[i][t] + 1) / (nt + V))
                if lp > ln:
                    pred.add(i)
            g = set(gold)
            tp += len(pred & g); fp += len(pred - g); fn += len(g - pred)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    print(f"   Micro-F1 cua baseline tra tu dien: {f1:.1%}  (P={p:.1%} R={r:.1%})")
    print("   -> DAT (data khong giai duoc bang khop tu khoa)" if f1 < 0.70
          else "   -> CANH BAO: data van de bi 'giai' bang tu khoa, nen sinh lai")

    # ---------- 3. Cấu trúc ----------
    print("\n" + "=" * 70)
    print("3. KIEM TRA LAI BENH CUA BO v3")
    by_n = defaultdict(list)
    for q, labs in gen:
        by_n[len(labs)].append(q)
    print(f"   {'So nhan':10} {'n':>6} {'% tu noi cam':>14} {'do dai TB (tu)':>16}")
    for k in sorted(by_n):
        qs = by_n[k]
        conn = sum(1 for q in qs if any(strip_accents(c).lower() in strip_accents(q).lower()
                                        for c in BANNED_CONNECTORS)) / len(qs)
        ln = statistics.mean(len(q.split()) for q in qs)
        print(f"   {k} nhan{'':4} {len(qs):>6} {conn:>13.1%} {ln:>16.1f}")

    slices = Counter()
    for line in args.generated.open(encoding="utf-8"):
        slices[json.loads(line)["slice"]] += 1
    print(f"\n   Phan bo dang cau: {dict(slices.most_common())}")


if __name__ == "__main__":
    main()
