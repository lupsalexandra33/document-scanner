import argparse
import json
import os
import re

import cv2
from easyocr import Reader

import mrz as mrz_module
import preprocessing
import validation

reader = Reader(["en"], gpu=False)

DEFAULT_VARIANT = "warped"

DATE_PATTERN = r'\d{2}[.\-]\d{2}[.\-]\d{4}'
DOCNUM_PATTERN = r'[A-Z0-9]\d{8}'
CNP_PATTERN = r'\d{13}'
DOCNUM_MIN_CONFIDENCE = 0.4


def parse_year(date_str):
    return int(date_str[-4:])


def assign_dates(dates):
    # map an unlabelled list of dates onto birth / issue / expiry.
    birth = issue = expiry = None
    if len(dates) == 1:
        birth = dates[0]
    elif len(dates) == 2:
        gap = parse_year(dates[1]) - parse_year(dates[0])
        if gap <= 15:
            issue, expiry = dates[0], dates[1]
        else:
            birth, expiry = dates[0], dates[1]
    elif len(dates) >= 3:
        birth, issue, expiry = dates[0], dates[1], dates[2]
    return birth, issue, expiry

def process_document(image_path, variant=DEFAULT_VARIANT):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    quad, method = preprocessing.detect_document_improved(image)
    prepared = preprocessing.PREPROCESSING_VARIANTS[variant](image, quad)
    results = reader.readtext(prepared)

    dates, doc_numbers, cnp, document_type = [], [], None, None
    min_confidence = 1.0
    texts = [text for (_, text, _) in results]

    for (bbox, text, prob) in results:
        date_match = re.search(DATE_PATTERN, text)
        if date_match:
            dates.append(date_match.group())
            min_confidence = min(min_confidence, prob)

        if prob > DOCNUM_MIN_CONFIDENCE:
            docnum_match = re.search(DOCNUM_PATTERN, text)
            if docnum_match:
                doc_numbers.append(docnum_match.group())

        cnp_match = re.search(CNP_PATTERN, text)
        if cnp_match:
            cnp = cnp_match.group()

        upper = text.upper()
        if "ROMANIA" in upper:
            document_type = "rou_drvlic"
        elif "ALBANIAN" in upper:
            document_type = "alb_id"

    dates = sorted(set(dates), key=parse_year)
    date_of_birth, date_of_issue, date_of_expiry = assign_dates(dates)
    last_name = first_name = None

    fields = {
        "last_name": last_name,
        "first_name": first_name,
        "date_of_birth": date_of_birth,
        "place_of_birth": None,
        "date_of_issue": date_of_issue,
        "date_of_expiry": date_of_expiry,
        "issued_by": None,
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
                fields[name] = value
            elif fields[name] != value:
                mrz_conflicts.append(
                    f"{name}: printed '{fields[name]}' vs MRZ '{value}'")

    ocr_confidence = min_confidence if dates else None
    validation_block = validation.validate(fields, ocr_confidence)

    if len(dates) < 3:
        validation_block["warnings"].append(
            f"only {len(dates)} dates extracted out of 3 expected")
    validation_block["warnings"] += mrz_conflicts

    if mrz_data:
        failed = [k for k, v in mrz_data["checks"].items() if v is False]
        if failed:
            validation_block["warnings"].append(
                f"MRZ check digit failed for: {', '.join(failed)}")

    return {
        "document_type": document_type,
        "fields": fields,
        "validation": validation_block,
        "mrz": {
            "found": mrz_data is not None,
            "format": mrz_data["format"] if mrz_data else None,
            "check_digits": mrz_data["checks"] if mrz_data else None,
        },
        "processing": {
            "detection_method": method,
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

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = args.out or f"outputs/{base}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out_path}")
