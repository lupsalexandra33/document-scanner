"""
run_donut_testset.py — run Donut over the fixed test set, one process per image.

    python run_donut_testset.py                      all 20, all fields
    python run_donut_testset.py --limit 5            a subset
    python run_donut_testset.py --fields last_name date_of_birth
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="testset.json")
    parser.add_argument("--out-dir", default="outputs_donut_week5")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N images of the test set")
    parser.add_argument("--fields", nargs="*", default=None,
                        help="restrict which fields Donut is asked about")
    args = parser.parse_args()

    if not os.path.exists(args.testset):
        print(f"{args.testset} not found — run build_testset.py first")
        return

    with open(args.testset, encoding="utf-8") as f:
        files = json.load(f)["files"]
    if args.limit:
        files = files[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    fields_note = ", ".join(args.fields) if args.fields else "all 8"
    per_image = 25 * (len(args.fields) if args.fields else 8)
    print(f"{len(files)} images, fields: {fields_note}")
    print(f"estimated {per_image}s per image, "
          f"{per_image * len(files) / 60:.0f} min total\n")

    done = failed = skipped = 0
    started = time.time()

    for index, image_path in enumerate(files, 1):
        stem = os.path.basename(image_path).rsplit(".", 1)[0]
        out_path = os.path.join(args.out_dir, f"{stem}.json")

        if os.path.exists(out_path):
            skipped += 1
            continue

        print(f"[{index}/{len(files)}] {stem[:44]}", flush=True)
        command = [sys.executable, "donut_pipeline.py", image_path,
                   "--out", out_path]
        if args.fields:
            command += ["--fields"] + args.fields

        result = subprocess.run(command, capture_output=True)
        if result.returncode == 0:
            done += 1
        else:
            failed += 1
            tail = result.stderr.decode()[-200:].strip()
            print(f"    failed: {tail}", flush=True)

    elapsed = (time.time() - started) / 60
    print(f"\ndone {done}, failed {failed}, already present {skipped} "
          f"({elapsed:.0f} min)")
    print(f"results in {args.out_dir}/")


if __name__ == "__main__":
    main()
