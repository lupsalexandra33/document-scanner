"""
    python results_matrix.py
    python results_matrix.py --markdown        for pasting into the report
"""
import argparse
import json
from collections import defaultdict

SYMBOLS = {"correct": "+", "wrong": "x", "missing": ".", "reject": "-"}

FIELD_ORDER = [
    "last_name", "first_name", "date_of_birth", "place_of_birth",
    "date_of_issue", "date_of_expiry", "issued_by", "document_number",
]


def short_name(filename, width=12):
    stem = filename.split("-ID_CARD")[0].split("-PP_TD3")[0]
    return stem[:width]


def print_matrix(rows, pipeline, markdown=False):
    present = [f for f in FIELD_ORDER
               if any(f in r.get(pipeline, {}).get("outcomes", {}) for r in rows)]
    if not present:
        print(f"no results for '{pipeline}'")
        return

    headers = [f[:9] for f in present]
    if markdown:
        print(f"| document | side | " + " | ".join(headers) + " |")
        print("|" + "---|" * (len(headers) + 2))
    else:
        print(f"{'document':13s} {'side':5s} " + " ".join(f"{h:>9s}" for h in headers))

    for row in rows:
        outcomes = row.get(pipeline, {}).get("outcomes", {})
        cells = [SYMBOLS.get(outcomes.get(f), " ") for f in present]
        name, side = short_name(row["file"]), row["side"][:5]
        if markdown:
            print(f"| {name} | {side} | " + " | ".join(cells) + " |")
        else:
            print(f"{name:13s} {side:5s} " +
                  " ".join(f"{c:>9s}" for c in cells))

    # column summary: which fields fail systematically
    print()
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for field, outcome in row.get(pipeline, {}).get("outcomes", {}).items():
            totals[field][outcome] += 1
    print(f"{'field':20s} {'correct':>8s} {'wrong':>7s} {'missing':>8s} {'reject':>7s}")
    for field in present:
        counts = totals[field]
        print(f"{field:20s} {counts['correct']:>8d} {counts['wrong']:>7d} "
              f"{counts['missing']:>8d} {counts['reject']:>7d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_evaluation.json")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]

    # score_outputs.py writes outcomes at the top level (one pipeline);
    # evaluate_final.py nests them per pipeline. Support both.
    if rows and "outcomes" in rows[0]:
        rows = [{**r, "single": {"outcomes": r["outcomes"]}} for r in rows]
        pipelines = ["single"]
    else:
        pipelines = [p for p in ("classic", "donut") if any(p in r for r in rows)]

    size = data.get("documents", len(rows))
    print(f"test set: {size} documents")
    print("legend: + correct   x wrong   . missing   - reject\n")

    for pipeline in pipelines:
        label = "classic" if pipeline == "single" else pipeline
        print(f"{'=' * 20} {label} {'=' * 20}")
        print_matrix(rows, pipeline, args.markdown)
        print()


if __name__ == "__main__":
    main()
