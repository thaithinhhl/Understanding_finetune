#!/usr/bin/env python3
"""Chấm điểm prediction của các task understanding theo metrics_understanding.md.

Tách rời khỏi bước inference: đọc file prediction đã sinh sẵn, không cần GPU.
"""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

GROUPS = ("C1", "U1", "U2", "U3")
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "question": str,
    "intent": str,
    "concepts": list,
    "asks_for_definition": bool,
    "scope": list,
    "user_facts": list,
    "asserts_premise": bool,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="File test gốc, để lấy slice và gold.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default=None, help="Tên model hiển thị trong báo cáo.")
    return parser.parse_args()


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.strip().split()).lower()


def concepts_set(value: Any) -> set[str]:
    return {norm(c) for c in value} if isinstance(value, list) else set()


def scope_set(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        return set()
    return {
        (norm(i.get("kind")), norm(i.get("value")))
        for i in value
        if isinstance(i, dict)
    }


def facts_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {norm(i.get("text") if isinstance(i, dict) else i) for i in value}


def micro_prf(pairs: list[tuple[set, set]]) -> dict[str, Any]:
    """Micro-average theo mục 1.1, gồm cả hai trường hợp mẫu số bằng 0."""
    tp = fp = fn = 0
    for pred, gold in pairs:
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    out: dict[str, Any] = {"tp": tp, "fp": fp, "fn": fn, "n_samples": len(pairs)}
    if tp + fn == 0:
        # Trường hợp 1: gold không có positive nào -> recall không xác định, không báo F1.
        out.update(
            precision=(tp / (tp + fp)) if tp + fp else None,
            recall=None,
            f1=None,
            note="TP+FN=0: ground truth không có positive, không báo F1 (xem mục 1.1)",
        )
        return out
    if tp + fp == 0:
        # Trường hợp 2: model không sinh positive nào.
        out.update(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            note="TP+FP=0: model không sinh phần tử nào cho field này",
        )
        return out
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    out.update(precision=precision, recall=recall, f1=f1, note=None)
    return out


def restraint_rate(pairs: list[tuple[set, set]]) -> dict[str, Any]:
    empty_gold = [(p, g) for p, g in pairs if not g]
    if not empty_gold:
        return {"rate": None, "n": 0}
    ok = sum(1 for p, _ in empty_gold if not p)
    return {"rate": ok / len(empty_gold), "n": len(empty_gold), "correct": ok}


def macro_f1(pred_labels: list[Any], gold_labels: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold_labels))
    per_label = {}
    for label in labels:
        tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_label[label] = {"f1": f1, "precision": precision, "recall": recall, "support": tp + fn}
    return {
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(per_label) if per_label else None,
        "per_label": per_label,
    }


def binary_f1(preds: list[Any], golds: list[bool]) -> dict[str, Any]:
    tp = sum(1 for p, g in zip(preds, golds) if p is True and g is True)
    fp = sum(1 for p, g in zip(preds, golds) if p is True and g is not True)
    fn = sum(1 for p, g in zip(preds, golds) if p is not True and g is True)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "baseline_always_false": sum(1 for g in golds if g is False) / len(golds),
    }


def schema_valid(pred: dict[str, Any] | None) -> bool:
    if not isinstance(pred, dict):
        return False
    for field, expected in REQUIRED_FIELDS.items():
        if field not in pred or not isinstance(pred[field], expected):
            return False
    return True


def current_turn(user_content: str) -> str:
    marker = "Current turn:"
    return user_content.split(marker, 1)[1].strip() if marker in user_content else user_content.strip()


def by_slice(items: list[dict], key_fn) -> dict[str, Any]:
    out = {}
    for slice_name in sorted({i["slice"] for i in items}):
        subset = [i for i in items if i["slice"] == slice_name]
        out[slice_name] = {"n": len(subset), "value": key_fn(subset)}
    return out


def score_group(group: str, items: list[dict]) -> dict[str, Any]:
    """items: mỗi phần tử gồm pred (dict|None), gold (dict), slice (str), user (str)."""
    n = len(items)
    report: dict[str, Any] = {"n": n}

    # --- Mục 6: metric định dạng dùng chung ---
    report["format"] = {
        "json_valid_rate": sum(1 for i in items if i["pred"] is not None) / n,
        "schema_valid_rate": sum(1 for i in items if schema_valid(i["pred"])) / n,
    }

    if group == "U1":
        preds = [i["pred"].get("intent") if i["pred"] else None for i in items]
        golds = [i["gold"]["intent"] for i in items]
        accuracy = sum(1 for p, g in zip(preds, golds) if p == g) / n
        majority = Counter(golds).most_common(1)[0]
        report["primary"] = {
            "name": "Intent Accuracy",
            "value": accuracy,
            "baseline": {"strategy": f"luôn trả '{majority[0]}'", "value": majority[1] / n},
        }
        report["supplementary"] = {"intent_macro_f1": macro_f1(preds, golds)}
        report["by_slice"] = by_slice(
            [{**i, "slice": "clean" if i["slice"] == "clean" else "boundary"} for i in items],
            lambda sub: sum(
                1 for i in sub
                if (i["pred"].get("intent") if i["pred"] else None) == i["gold"]["intent"]
            ) / len(sub),
        )

    elif group == "U2":
        pairs = [(concepts_set(i["pred"].get("concepts")) if i["pred"] else set(),
                  concepts_set(i["gold"].get("concepts"))) for i in items]
        exact = sum(1 for p, g in pairs if p == g) / n
        baseline_empty = sum(1 for _, g in pairs if not g) / n
        report["primary"] = {
            "name": "Concept Exact Match Accuracy",
            "value": exact,
            "baseline": {"strategy": "luôn trả []", "value": baseline_empty},
        }
        report["supplementary"] = {
            "concept_micro": micro_prf(pairs),
            "concept_restraint_rate": restraint_rate(pairs),
        }
        report["by_slice"] = by_slice(
            items,
            lambda sub: sum(
                1 for i in sub
                if (concepts_set(i["pred"].get("concepts")) if i["pred"] else set())
                == concepts_set(i["gold"].get("concepts"))
            ) / len(sub),
        )

    elif group == "U3":
        def struct_ok(item: dict) -> bool:
            pred, gold = item["pred"], item["gold"]
            if not pred:
                return False
            return (scope_set(pred.get("scope")) == scope_set(gold.get("scope"))
                    and facts_set(pred.get("user_facts")) == facts_set(gold.get("user_facts"))
                    and pred.get("asserts_premise") == gold.get("asserts_premise"))

        def all_empty(gold: dict) -> bool:
            return not gold.get("scope") and not gold.get("user_facts") and not gold.get("asserts_premise")

        extraction = [i for i in items if not all_empty(i["gold"])]
        restraint = [i for i in items if all_empty(i["gold"])]
        report["primary"] = {
            "structured_em_overall": {
                "value": sum(1 for i in items if struct_ok(i)) / n,
                "n": n,
                "baseline": {
                    "strategy": "luôn trả scope=[], user_facts=[], asserts_premise=false",
                    "value": len(restraint) / n,
                },
            },
            "structured_em_extraction": {
                "value": sum(1 for i in extraction if struct_ok(i)) / len(extraction),
                "n": len(extraction),
            },
            "structured_restraint_rate": {
                "value": sum(1 for i in restraint if struct_ok(i)) / len(restraint),
                "n": len(restraint),
            },
        }
        scope_pairs = [(scope_set(i["pred"].get("scope")) if i["pred"] else set(),
                        scope_set(i["gold"].get("scope"))) for i in items]
        facts_pairs = [(facts_set(i["pred"].get("user_facts")) if i["pred"] else set(),
                        facts_set(i["gold"].get("user_facts"))) for i in items]
        report["supplementary"] = {
            "scope_micro": micro_prf(scope_pairs),
            "scope_restraint_rate": restraint_rate(scope_pairs),
            "user_facts_micro": micro_prf(facts_pairs),
            "premise": binary_f1(
                [i["pred"].get("asserts_premise") if i["pred"] else None for i in items],
                [bool(i["gold"].get("asserts_premise")) for i in items],
            ),
        }
        report["by_slice"] = by_slice(items, lambda sub: sum(1 for i in sub if struct_ok(i)) / len(sub))

    elif group == "C1":
        exact = sum(
            1 for i in items
            if i["pred"] and norm(i["pred"].get("question", "")) == norm(i["gold"]["question"])
        ) / n
        copy_baseline = sum(
            1 for i in items if norm(current_turn(i["user"])) == norm(i["gold"]["question"])
        ) / n
        report["primary"] = {
            "name": "Question Semantic Accuracy",
            "value": None,
            "note": "Cần LLM judge riêng theo mục 5.2, chưa chấm trong bước này.",
        }
        report["supplementary"] = {
            "question_exact_match": {
                "value": exact,
                "baseline": {"strategy": "chép nguyên câu hỏi lượt hiện tại", "value": copy_baseline},
            }
        }
        report["by_slice"] = by_slice(
            items,
            lambda sub: sum(
                1 for i in sub
                if i["pred"] and norm(i["pred"].get("question", "")) == norm(i["gold"]["question"])
            ) / len(sub),
        )

    return report


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if isinstance(value, float) else str(value)


def main() -> None:
    args = parse_args()

    source = {}
    for line in args.data.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            source[row["id"]] = row

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    items_by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for result in payload["results"]:
        row = source[result["id"]]
        items_by_group[row["group"]].append({
            "id": result["id"],
            "pred": result.get("prediction"),
            "gold": json.loads(row["messages"][2]["content"]),
            "slice": row["slice"],
            "user": row["messages"][1]["content"],
        })

    label = args.label or payload.get("adapter") or payload.get("model")
    report = {
        "label": label,
        "model": payload.get("model"),
        "adapter": payload.get("adapter"),
        "precision": payload.get("precision"),
        "predictions_file": str(args.predictions),
        "groups": {g: score_group(g, items) for g, items in items_by_group.items() if items},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"### Báo cáo metric — {label}\n")
    for group, r in report["groups"].items():
        print(f"[{group}]  n={r['n']}  json_valid={fmt(r['format']['json_valid_rate'])}  "
              f"schema_valid={fmt(r['format']['schema_valid_rate'])}")
        primary = r["primary"]
        if group == "U3":
            for key, value in primary.items():
                base = f"   (baseline {fmt(value['baseline']['value'])})" if "baseline" in value else ""
                print(f"    {key:28} {fmt(value['value'])}  n={value['n']}{base}")
        else:
            base = f"   (baseline {fmt(primary['baseline']['value'])})" if primary.get("baseline") else ""
            print(f"    {primary['name']:28} {fmt(primary['value'])}{base}")
            if primary.get("note"):
                print(f"      -> {primary['note']}")
        for key, value in r["supplementary"].items():
            if isinstance(value, dict) and "f1" in value:
                extra = f"  TP={value['tp']} FP={value['fp']} FN={value['fn']}"
                print(f"    {key:28} F1={fmt(value['f1'])}  P={fmt(value['precision'])}  R={fmt(value['recall'])}{extra}")
                if value.get("note"):
                    print(f"      -> {value['note']}")
            elif isinstance(value, dict) and "macro_f1" in value:
                print(f"    {key:28} {fmt(value['macro_f1'])}")
                for lab, stats in sorted(value["per_label"].items(), key=lambda kv: -kv[1]["support"]):
                    print(f"        {lab:22} F1={fmt(stats['f1'])}  n={stats['support']}")
            elif isinstance(value, dict) and "rate" in value:
                print(f"    {key:28} {fmt(value['rate'])}  n={value['n']}")
            elif isinstance(value, dict) and "value" in value:
                base = f"   (baseline {fmt(value['baseline']['value'])})" if value.get("baseline") else ""
                print(f"    {key:28} {fmt(value['value'])}{base}")
        print(f"    by_slice: " + "  ".join(
            f"{k}={fmt(v['value'])}(n={v['n']})" for k, v in r["by_slice"].items()))
        print()
    print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
