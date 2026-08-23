import argparse
import json
import os
import re

import cv2
from easyocr import Reader

import mrz as mrz_module
import preprocessing
import validation

# load the OCR engine once, it is slow to start, so keep it at module level
reader = Reader(["en"], gpu=False)

# which preprocessed image to send to OCR
DEFAULT_VARIANT = "warped"

DATE_PATTERN = r'\d{2}[.\-]\d{2}[.\-]\d{4}'   # 10.08.2018 or 15-06-2014
DOCNUM_PATTERN = r'[A-Z0-9]\d{8}'             # letter/digit + 8 digits
CNP_PATTERN = r'\d{13}'                       # Romanian personal number
DOCNUM_MIN_CONFIDENCE = 0.4                   # ignore low-confidence matches


def parse_year(date_str):
    # year as an int; takes the last 4 characters so both '.' and '-' work
    return int(date_str[-4:])


def assign_dates(dates):
    # map an unlabelled list of dates onto birth / issue / expiry
    birth = issue = expiry = None
    if len(dates) == 1:
        # a lone date is most likely the birth date
        birth = dates[0]
    elif len(dates) == 2:
        gap = parse_year(dates[1]) - parse_year(dates[0])
        if gap <= 15:
            # dates close together: a document issued and expiring years apart
            issue, expiry = dates[0], dates[1]
        else:
            # dates far apart: the old one is a birth date
            birth, expiry = dates[0], dates[1]
    elif len(dates) >= 3:
        # oldest is birth, then issue, then expiry
        birth, issue, expiry = dates[0], dates[1], dates[2]
    return birth, issue, expiry


def extract_names_albanian(results):
    candidates = []
    for (bbox, text, prob) in results:
        y = int(bbox[0][1])   # vertical position; smaller means higher up
        # names are confident, near the top, and letters only (not dates)
        if prob > 0.8 and y < 160 and text.isalpha():
            candidates.append((y, text))
    candidates.sort()   # top to bottom
    last = candidates[0][1] if candidates else None
    first = candidates[1][1] if len(candidates) >= 2 else None
    return last, first


def process_document(image_path, variant=DEFAULT_VARIANT):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    # locate the document; returns None if nothing plausible was found
    quad, method = preprocessing.detect_document_improved(image)
    # prepare the image for OCR (falls back to the original if quad is None)
    prepared = preprocessing.PREPROCESSING_VARIANTS[variant](image, quad)
    # each result is (bounding box, text, confidence)
    results = reader.readtext(prepared)

    dates, doc_numbers, cnp, document_type = [], [], None, None
    min_confidence = 1.0
    texts = [text for (_, text, _) in results]

    # scan every text fragment once, testing all patterns against it
    for (bbox, text, prob) in results:
        date_match = re.search(DATE_PATTERN, text)
        if date_match:
            dates.append(date_match.group())
            # track the weakest reading, as an overall confidence indicator
            min_confidence = min(min_confidence, prob)

        # the document number and parts of the CNP look alike, so the
        # confidence score is used to tell them apart
        if prob > DOCNUM_MIN_CONFIDENCE:
            docnum_match = re.search(DOCNUM_PATTERN, text)
            if docnum_match:
                doc_numbers.append(docnum_match.group())

        cnp_match = re.search(CNP_PATTERN, text)
        if cnp_match:
            cnp = cnp_match.group()

        # document type is guessed from keywords printed on the document
        upper = text.upper()
        if "ROMANIA" in upper:
            document_type = "rou_drvlic"
        elif "ALBANIAN" in upper:
            document_type = "alb_id"

    # deduplicate and sort by year before assigning dates to fields
    dates = sorted(set(dates), key=parse_year)
    date_of_birth, date_of_issue, date_of_expiry = assign_dates(dates)

    last_name = first_name = None
    # position-based name rules only hold for the layout they were written for
    if document_type == "alb_id":
        last_name, first_name = extract_names_albanian(results)

    fields = {
        "last_name": last_name,
        "first_name": first_name,
        "date_of_birth": date_of_birth,
        "place_of_birth": None,        # not covered by any rule yet
        "date_of_issue": date_of_issue,
        "date_of_expiry": date_of_expiry,
        "issued_by": None,             # not covered by any rule yet
        "document_number": doc_numbers[0] if doc_numbers else None,
        "personal_number": cnp,
    }

    # MRZ
    mrz_lines = mrz_module.find_mrz_lines(texts)
    mrz_data = mrz_module.parse(mrz_lines) if mrz_lines else None
    mrz_conflicts = []

    if mrz_data:
        for name, value in mrz_module.to_fields(mrz_data).items():
            if value is None:
                continue
            if fields.get(name) is None:
                fields[name] = value          # fill a gap
            elif fields[name] != value:
                # never overwrite: report the disagreement instead
                mrz_conflicts.append(
                    f"{name}: printed '{fields[name]}' vs MRZ '{value}'")

    ocr_confidence = min_confidence if dates else None
    # run every validation rule (see validation.py)
    validation_block = validation.validate(fields, ocr_confidence)

    if len(dates) < 3:
        validation_block["warnings"].append(
            f"only {len(dates)} dates extracted out of 3 expected")
    validation_block["warnings"] += mrz_conflicts

    if mrz_data:
        # a failed check digit means the MRZ was misread by OCR
        failed = [k for k, v in mrz_data["checks"].items() if v is False]
        if failed:
            validation_block["warnings"].append(
                f"MRZ check digit failed for: {', '.join(failed)}")

    return {
        "document_type": document_type,
        "fields": fields,
        "validation": validation_block,
        # raw OCR fragments, kept so evaluation can measure the MRZ reading
        # stage separately from the extraction rules
        "ocr_texts": texts,
        "mrz": {
            "found": mrz_data is not None,
            "format": mrz_data["format"] if mrz_data else None,
            "check_digits": mrz_data["checks"] if mrz_data else None,
        },
        "processing": {
            "detection_method": method,            # "contour", "minarea" or null
            "document_detected": quad is not None,
            "preprocessing_variant": variant,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=list(preprocessing.PREPROCESSING_VARIANTS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = process_document(args.image, variant=args.variant)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # save next to the other outputs, named after the input image
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = args.out or f"outputs/{base}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out_path}")
