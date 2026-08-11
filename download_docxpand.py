"""
Use --parts to control how far to go:

    python download_docxpand.py     50 per class, parts 00-05
    python download_docxpand.py --per-class 20      fewer images
    python download_docxpand.py --parts 12      all classes (streams 17 GB)
    python download_docxpand.py --labels     also fetch the ground truth
"""
import argparse
import os
import shutil
import subprocess
import sys

BASE_URL = ("https://github.com/QuickSign/docxpand/releases/download/v1.0.0/"
            "DocXPand-25k.tar.gz.")

OUT_DIR = "data/docxpand"

CLASSES = [
    "ID_CARD_TD1_A", "ID_CARD_TD1_B", "ID_CARD_TD2_A", "ID_CARD_TD2_B",
    "PP_TD3_A", "PP_TD3_B", "PP_TD3_C", "RP_CARD_TD1", "RP_CARD_TD2",
]


def stream_extract(parts, pattern, dest):
    # stream the given archive parts through tar, extracting only `pattern`
    urls = [f"{BASE_URL}{i:02d}" for i in parts]
    os.makedirs(dest, exist_ok=True)

    curl = subprocess.Popen(["curl", "-sL", *urls], stdout=subprocess.PIPE)
    tar = subprocess.Popen(
        ["tar", "xzf", "-", "-C", dest, "--wildcards", pattern],
        stdin=curl.stdout, stderr=subprocess.DEVNULL,
    )
    curl.stdout.close()
    tar.wait()
    curl.terminate()


def trim_to_limit(per_class):
    # keep at most `per_class` images per document class, delete the rest
    docs_dir = os.path.join(OUT_DIR, "DocXPand-25k", "documents")
    if not os.path.isdir(docs_dir):
        return {}

    kept = {}
    for class_name in sorted(os.listdir(docs_dir)):
        class_dir = os.path.join(docs_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(os.listdir(class_dir))
        for extra in files[per_class:]:
            os.remove(os.path.join(class_dir, extra))
        kept[class_name] = min(len(files), per_class)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=50,
                        help="how many images to keep per document class")
    parser.add_argument("--parts", type=int, default=6,
                        help="how many archive parts to stream (more parts = "
                             "more classes; 12 = all of them)")
    parser.add_argument("--labels", action="store_true",
                        help="also download the ground truth labels (~200 MB, "
                             "found in the first part)")
    args = parser.parse_args()

    if args.labels:
        print("fetching ground truth labels (streams ~1.5 GB, writes ~200 MB)...")
        stream_extract([0], "DocXPand-25k/labels/DocXPand-25k.json", OUT_DIR)
        label_path = os.path.join(OUT_DIR, "DocXPand-25k", "labels",
                                  "DocXPand-25k.json")
        print("  ok" if os.path.exists(label_path) else "  labels not found")

    print(f"streaming parts 00-{args.parts - 1:02d} "
          f"(~{args.parts * 1.5:.0f} GB through the network, archives are not "
          f"stored)...")
    print("a 'tar: Unexpected EOF' message at the end is expected — we stop "
          "the stream on purpose.\n")

    stream_extract(range(args.parts), "DocXPand-25k/documents/*", OUT_DIR)

    kept = trim_to_limit(args.per_class)
    if not kept:
        print("no images extracted — check your connection and try again")
        return

    print("\nimages kept per class:")
    for class_name in CLASSES:
        count = kept.get(class_name, 0)
        marker = "" if count else "   (needs more parts)"
        print(f"  {class_name:16s} {count:4d}{marker}")

    total = sum(kept.values())
    docs_dir = os.path.join(OUT_DIR, "DocXPand-25k", "documents")
    size_mb = sum(os.path.getsize(os.path.join(root, f))
                  for root, _, files in os.walk(docs_dir)
                  for f in files) / 1024 / 1024
    print(f"\ntotal: {total} images, {size_mb:.0f} MB in {docs_dir}/")


if __name__ == "__main__":
    main()