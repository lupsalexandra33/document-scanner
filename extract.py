from easyocr import Reader
import cv2
import scanner
import re
import json

# load the OCR reader
reader = Reader(["en"], gpu=False)

def parse_year(date_str):
    return int(date_str[-4:])

def is_valid_date(date_str):
    # split on either '.' or '-'
    parts = re.split(r'[.\-]', date_str)
    if len(parts) != 3:
        return False
    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
    if not (1 <= day <= 31):
        return False
    if not (1 <= month <= 12):
        return False
    if not (1900 <= year <= 2100):
        return False
    return True

def extract_names_albanian(results):
    candidates = []
    for (bbox, text, prob) in results:
        y = int(bbox[0][1])
        if prob > 0.8 and y < 160 and text.isalpha():
            candidates.append((y, text))

    candidates.sort()

    last_name = candidates[0][1] if len(candidates) >= 1 else None
    first_name = candidates[1][1] if len(candidates) >= 2 else None
    return last_name, first_name

def process_document(image_path):
    # read the image and run it through the scanner
    image = cv2.imread(image_path)
    quad = scanner.detect_document(image)
    warped = scanner.four_point_transform(image, quad) if quad is not None else image

    # OCR
    results = reader.readtext(warped)

    # extraction patterns
    date_pattern = r'\d{2}[.\-]\d{2}[.\-]\d{4}'
    docnum_pattern = r'[A-Z0-9]\d{8}'
    # Romanian CNP = 13 digits
    cnp_pattern = r'\d{13}'

    dates = []
    doc_numbers = []
    cnp = None
    document_type = None
    min_confidence = 1.0

    for (bbox, text, prob) in results:
        # dates
        date_match = re.search(date_pattern, text)
        if date_match:
            dates.append(date_match.group())
            min_confidence = min(min_confidence, prob)

        # document number
        if prob > 0.4:
            docnum_match = re.search(docnum_pattern, text)
            if docnum_match:
                doc_numbers.append(docnum_match.group())

        # CNP (13 digits)
        cnp_match = re.search(cnp_pattern, text)
        if cnp_match:
            cnp = cnp_match.group()

        # document type
        if "ROMANIA" in text.upper():
            document_type = "rou_drvlic"
        elif "ALBANIAN" in text.upper():
            document_type = "alb_id"

    # assign the dates to fields (by year)
    dates = sorted(set(dates), key=parse_year)

    date_of_birth = None
    date_of_issue = None
    date_of_expiry = None

    if len(dates) == 1:
        date_of_birth = dates[0]
    elif len(dates) == 2:
        gap = parse_year(dates[1]) - parse_year(dates[0])
        if gap <= 15:
            date_of_issue = dates[0]
            date_of_expiry = dates[1]
        else:
            date_of_birth = dates[0]
            date_of_expiry = dates[1]
    elif len(dates) >= 3:
        date_of_birth = dates[0]
        date_of_issue = dates[1]
        date_of_expiry = dates[2]

    # names, only for the Albanian ID
    last_name = None
    first_name = None
    if document_type == "alb_id":
        last_name, first_name = extract_names_albanian(results)

    # JSON structure
    output = {
        "document_type": document_type,
        "fields": {
            "last_name": last_name,
            "first_name": first_name,
            "date_of_birth": date_of_birth,
            "place_of_birth": None,
            "date_of_issue": date_of_issue,
            "date_of_expiry": date_of_expiry,
            "issued_by": None,
            "document_number": doc_numbers[0] if doc_numbers else None,
            "personal_number": cnp,
        },
        "validation": {
            "missing_fields": [],
            "warnings": [],
            "ocr_confidence": round(min_confidence, 2) if dates else None,
        },
    }

    # missing fields
    for field_name, value in output["fields"].items():
        if value is None:
            output["validation"]["missing_fields"].append(field_name)

    # date format, check each extracted date is a real calendar date
    for field in ["date_of_birth", "date_of_issue", "date_of_expiry"]:
        value = output["fields"][field]
        if value is not None and not is_valid_date(value):
            output["validation"]["warnings"].append(f"{field}: '{value}' is not a valid date")

    # CNP length
    if cnp is not None and len(cnp) != 13:
        output["validation"]["warnings"].append(f"personal_number: expected 13 digits, got {len(cnp)}")

    # how many dates were found vs expected
    if len(dates) < 3:
        output["validation"]["warnings"].append(f"only {len(dates)} dates extracted out of 3 expected")

    return output

if __name__ == "__main__":
    result = process_document("data/38_rou_drvlic/images/TA/TA38_25.tif")

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nsaved to output.json")