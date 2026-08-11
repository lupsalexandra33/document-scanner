# Document Scanner

A pipeline for reading identity documents: it locates the document in a photo,
corrects its perspective, runs OCR, parses the machine-readable zone where one
exists, and extracts the fields into a validated JSON.

## Datasets

Two sets are used, for different purposes.

**[DocXPand-25k](https://github.com/QuickSign/docxpand)** = synthetic identity
documents photographed on real backgrounds, with ground truth for both the
document corners and every printed field. This is the main evaluation set,
because it is the only one that makes the metrics measurable rather than
eyeballed.

```bash
python3 download_docxpand.py --labels --per-class 30 --parts 8
```

The full dataset is 17 GB in 12 archive parts. The script never stores them: it
streams the parts straight into `tar` and writes only the document images, so
disk usage stays in the tens of MB.

**[MIDV-500](https://doi.org/10.18287/2412-6179-2019-43-5-818-824)** = video
frames of identity document specimens under different capture conditions. Kept
as the robustness set, since it contains real degradation (motion blur, a hand
occluding the border, cluttered surfaces) that a rendered dataset does not
reproduce.

```bash
python3 download_dataset.py --hard
```

Both contain specimen documents only, with no real personal data. Neither is
stored in this repository.

## Project structure

```
document-scanner/
├── scanner.py               # geometric pipeline: detect, warp, enhance
├── preprocessing.py         # improved detector + preprocessing variants
├── extract.py               # main pipeline: image -> OCR -> fields -> JSON
├── mrz.py                   # machine-readable zone: parsing + check digits
├── validation.py            # validation rules (separate module)
├── docxpand.py              # reads the DocXPand ground truth
├── evaluate.py              # detection metrics on MIDV
├── evaluate_docxpand.py     # IoU + field accuracy against ground truth
├── compare_variants.py      # compares preprocessing variants for OCR
├── download_docxpand.py     # streams a DocXPand subset
├── download_dataset.py      # downloads the MIDV-500 subset
├── document_scanner.ipynb   # notebook: the geometric stage, step by step
├── annotation_format.json   # JSON schema for extracted fields
├── OCR_limitations.md       # OCR limitations observed in week 2
├── week3_report.md          # robustness, MRZ and measurement results
├── examples/                # before/after demonstration images
├── outputs/                 # sample JSON outputs
└── data/                    # datasets (git-ignored, created by the scripts)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Extract fields from one image:

```bash
python3 extract.py path/to/image.jpg
python3 extract.py path/to/image.jpg --variant raw     # choose a variant
```

Measure the pipeline:

```bash
python3 evaluate_docxpand.py --detection-only    # geometry only, fast
python3 evaluate_docxpand.py --limit 20          # full pipeline with OCR
python3 evaluate.py --detection-only             # detection on MIDV
```

## Results

**Document detection.** The week 1 detector looks for a contour that simplifies
to exactly four corners. Week 3 adds a rotated-rectangle fallback for when no
clean quadrilateral exists, plus two geometric gates (area and aspect ratio)
that reject implausible detections:

| set | baseline | improved |
|-----|----------|----------|
| MIDV hard conditions (14 images) | 2 / 14 (14%) | 11 / 14 (79%) |
| DocXPand (4555 images) | - | 2540 / 4555 (56%), mean IoU 0.619 |

The gates matter: without them a looser fallback reported 12/14, but several of
those "detections" were the whole frame rather than the document.

**Field extraction** on DocXPand: 31% of fields correct, 4% wrong, 65% missing.
The low error rate is deliberate — the pipeline prefers to report nothing over
reporting a wrong value. Most correct values come from the MRZ rather than from
positional rules.

**MRZ** parsing follows ICAO 9303 (TD1/TD2/TD3) and verifies check digits, so a
misread can be detected rather than assumed. Validated against ground truth, the
document number matched on all 6498 annotated documents.

Full results, including two negative findings — a false positive on a background
object, and perspective correction slightly *reducing* accuracy on DocXPand —
are in [week3_report.md](week3_report.md). OCR limitations observed earlier are
in [OCR_limitations.md](OCR_limitations.md).