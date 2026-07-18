# Document Scanner

A document scanning pipeline that turns a phone photo of an identity document
into a clean, "scanned"-looking image. It detects the document in the photo,
corrects its perspective, and enhances it (grayscale, contrast, denoise,
binarization).

## Dataset

This project uses the [MIDV-500](https://doi.org/10.18287/2412-6179-2019-43-5-818-824)
dataset of identity document specimens. The documents
are synthetic samples and contain no real personal data.

The full dataset is **not** included in this repository. Run the download script
to fetch a small subset (Albanian ID + Romanian driving licence, clean-background
conditions only):

```bash
python3 download_dataset.py
```

Sample images shown in `examples/` are included for demonstration only.

## Project structure

```
document-scanner/
├── scanner.py              # core pipeline: detect, warp, enhance
├── document_scanner.ipynb  # notebook with step-by-step visualizations
├── download_dataset.py     # downloads the MIDV-500 subset
├── annotation_format.json  # JSON for extracted fields
├── examples/               # before/after demonstration images
├── requirements.txt        # Python dependencies
└── data/                   # dataset (created by download_dataset.py)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `document_scanner.ipynb` and run the cells top to bottom.