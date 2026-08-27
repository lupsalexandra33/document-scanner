# Document Scanner

Two pipelines for reading identity documents, built to be compared.

**Pipeline 1 (classic)** locates the document, corrects its perspective, runs
OCR, parses the machine-readable zone where one exists, and applies rules to
extract the fields.

**Pipeline 2 (pretrained model)** hands the whole image to a document-
understanding model and reads the fields straight out of it, no detection, no
OCR, no rules.

Both write the same JSON schema, so their results can be scored side by side
against the same ground truth.

## Datasets

Two sets are used, for different purposes.

**[DocXPand-25k](https://github.com/QuickSign/docxpand)** - synthetic identity
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

**[MIDV-500](https://doi.org/10.18287/2412-6179-2019-43-5-818-824)** - video
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
├── extract.py               # pipeline 1: image -> OCR -> rules -> JSON
├── donut_pipeline.py        # pipeline 2: image -> pretrained model -> JSON
├── mrz.py                   # machine-readable zone: parsing + check digits
├── validation.py            # validation rules (separate module)
├── docxpand.py              # reads the DocXPand ground truth
├── evaluate.py              # detection metrics on MIDV
├── evaluate_docxpand.py     # IoU + field accuracy against ground truth
├── compare_variants.py      # compares preprocessing variants for OCR
├── compare_pipelines.py     # runs both pipelines and scores them together
├── build_testset.py         # draws the fixed, independently chosen test set
├── score_outputs.py         # scores saved outputs: correct/wrong/missing/reject
├── evaluate_final.py        # full evaluation, both pipelines
├── results_matrix.py        # document-by-field results grid
├── app.py                   # Streamlit demo: both pipelines, side by side
├── .streamlit/config.toml   # app theme
├── download_docxpand.py     # streams a DocXPand subset
├── download_dataset.py      # downloads the MIDV-500 subset
├── document_scanner.ipynb   # notebook: the geometric stage, step by step
├── annotation_format.json   # JSON schema for extracted fields
├── OCR_limitations.md       # OCR limitations observed in week 2
├── week3_report.md          # robustness, MRZ and measurement results
├── week4_report.md          # pretrained model results and comparison
├── week5_report.md          # comparison, evaluation and conclusions
├── final_report.md          # full write-up of the project
├── examples/                # before/after demonstration images
├── outputs/                 # sample JSON outputs (pipeline 1)
├── outputs_donut/           # sample JSON outputs (pipeline 2)
├── outputs_week5/           # pipeline 1 outputs over the fixed test set
└── data/                    # datasets (git-ignored, created by the scripts)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a machine without a GPU, install the CPU build of torch first, it is a much
smaller download than the default CUDA one:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Pipeline 2 downloads its model (~800 MB) on first use.

## Demo application

```bash
pip install streamlit
streamlit run app.py
```

Upload a document image, pick a preprocessing variant and a pipeline, and the
app shows each stage: the localised document, what OCR receives, the raw text,
the extracted JSON and the validation warnings. Both pipelines can be run side
by side.

Donut takes roughly 25 seconds per field on CPU and needs about 1 GB of memory,
so the sidebar limits which fields it is asked about. Running both pipelines at
once loads two models simultaneously, on a machine with 8 GB that can fail, so
run them separately if the app is killed.

Selecting the `raw` preprocessing variant skips geometric correction entirely.
On DocXPand that measured slightly *better* than the corrected version, so it is
a reasonable default for the demo.

## Usage

Extract fields from one image:

```bash
python3 extract.py path/to/image.jpg                   # pipeline 1
python3 extract.py path/to/image.jpg --variant raw     # choose a variant
python3 donut_pipeline.py path/to/image.jpg            # pipeline 2
```

Pipeline 2 asks the model one question per field, and each question is a
separate forward pass, about 25 seconds each on CPU. Use `--fields` to limit
which ones are asked.

Measure the pipeline:

```bash
python3 evaluate_docxpand.py --detection-only    # geometry only, fast
python3 evaluate_docxpand.py --limit 20          # full pipeline with OCR
python3 evaluate.py --detection-only             # detection on MIDV
python3 compare_pipelines.py --limit 3           # both pipelines, same images
```

The week 5 evaluation runs in three steps. Extraction is done one process per
image because running OCR inside a loop exhausts memory on an 8 GB machine;
scoring is then a separate, fast pass over the saved results.

```bash
python3 build_testset.py --size 20               # fixed sample, seeded
# extract one process per image, into outputs_week5/
python3 score_outputs.py --dir outputs_week5     # four-outcome scoring
python3 results_matrix.py                        # document-by-field grid
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
The low error rate is deliberate, the pipeline prefers to report nothing over
reporting a wrong value. Most correct values come from the MRZ rather than from
positional rules.

**MRZ** parsing follows ICAO 9303 (TD1/TD2/TD3) and verifies check digits, so a
misread can be detected rather than assumed. Validated against ground truth, the
document number matched on all 6498 annotated documents.

**Classic versus pretrained model**, on the same images and the same ground
truth:

Measured on four card fronts with high detection quality (IoU 0.84–0.95):

| pipeline | correct | wrong | missing | accuracy | avg time |
|----------|---------|-------|---------|----------|----------|
| classic (OCR + rules) | 6 | 2 | 12 | **30%** | 7 s |
| Donut (DocVQA) | 2 | **18** | 0 | 10% | 181 s |

The hand-written rules beat the pretrained model here, which is the opposite of
the expected result, and the failure behaviour is close to opposite too. The
classic pipeline either reads a document well or produces nothing at all, it
was wrong twice and silent twelve times. Donut left nothing empty, getting 18 of
20 fields wrong, because it has no way to decline a question.

Used without fine-tuning it reads the document correctly but attaches values to
the wrong fields, asked for a surname it returned the MRZ string, and asked for
a date of birth it returned the surname.

**Classic pipeline on a fixed, independently drawn test set** (20 documents,
104 fields, stratified by class and side, seeded):

| outcome | share |
|---------|-------|
| correct | 26% |
| wrong | 4% |
| missing | 54% |
| reject | 16% |

Three derived figures say more than the accuracy does: an **error rate of 4%**,
an **abstention rate of 70%**, and **87% precision when it does answer** (27 of
31). The pipeline is not so much accurate as cautious.

The fields split cleanly: 27 of the 31 correct values are dates, and names,
places, authorities and document numbers score zero across all 20 documents.
A date has a fixed pattern a regex matches in any layout; the others need to be
located on the page. Rule-based extraction works exactly as far as regular
expressions reach.

The full write-up is in [final_report.md](final_report.md). Per-week results are
in [week3_report.md](week3_report.md),
[week4_report.md](week4_report.md) and [week5_report.md](week5_report.md),
including several negative findings: a false positive on a background object,
perspective correction *reducing* extraction accuracy on DocXPand, and the MRZ
being read end to end on only 3 of 13 documents despite the parser itself being
correct. OCR limitations observed earlier are in
[OCR_limitations.md](OCR_limitations.md).