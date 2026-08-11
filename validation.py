import re

# Romanian CNP is exactly 13 digits
CNP_LENGTH = 13
# plausible range for any date printed on an identity document
MIN_YEAR, MAX_YEAR = 1900, 2100
# characters we expect in a name field
NAME_ALLOWED = re.compile(r'^[A-Za-zĂÂÎȘȚăâîșț\s\.\-]+$')


def split_date(date_str):
    # split 'dd.mm.yyyy' or 'dd-mm-yyyy' into (day, month, year) ints
    parts = re.split(r'[.\-]', date_str)
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def is_valid_date(date_str):
    """True if the string is a real calendar date in a plausible year range."""
    parts = split_date(date_str)
    if parts is None:
        return False
    day, month, year = parts
    return 1 <= day <= 31 and 1 <= month <= 12 and MIN_YEAR <= year <= MAX_YEAR

# individual rules:

def rule_missing_fields(fields):
    # report which fields were not extracted at all
    return [name for name, value in fields.items() if value is None]


def rule_date_format(fields):
    # every extracted date must be a real calendar date
    warnings = []
    for name in ("date_of_birth", "date_of_issue", "date_of_expiry"):
        value = fields.get(name)
        if value is not None and not is_valid_date(value):
            warnings.append(f"{name}: '{value}' is not a valid date")
    return warnings


def rule_date_order(fields):
    # birth must precede issue, and issue must precede expiry
    warnings = []

    def year_of(name):
        v = fields.get(name)
        if v is None or not is_valid_date(v):
            return None
        return split_date(v)[2]

    birth, issue, expiry = year_of("date_of_birth"), year_of("date_of_issue"), year_of("date_of_expiry")
    if birth is not None and issue is not None and birth >= issue:
        warnings.append("date_of_birth is not before date_of_issue")
    if issue is not None and expiry is not None and issue >= expiry:
        warnings.append("date_of_issue is not before date_of_expiry")
    return warnings


def rule_cnp_length(fields):
    # the Romanian personal number must be exactly 13 digits
    cnp = fields.get("personal_number")
    if cnp is None:
        return []
    if len(cnp) != CNP_LENGTH:
        return [f"personal_number: expected {CNP_LENGTH} digits, got {len(cnp)}"]
    if not cnp.isdigit():
        return ["personal_number: contains non-digit characters"]
    return []


def rule_invalid_characters(fields):
    # name fields should not contain digits or OCR noise symbols
    warnings = []
    for name in ("last_name", "first_name", "place_of_birth", "issued_by"):
        value = fields.get(name)
        if value is not None and not NAME_ALLOWED.match(value):
            warnings.append(f"{name}: '{value}' contains unexpected characters")
    return warnings


def rule_low_confidence(fields, ocr_confidence, threshold=0.4):
    # flag results the OCR engine itself was unsure about
    if ocr_confidence is not None and ocr_confidence < threshold:
        return [f"low OCR confidence ({ocr_confidence:.2f}) — fields may be unreliable"]
    return []


def validate(fields, ocr_confidence=None):
    # run every rule and return the validation block for the output JSON
    warnings = []
    warnings += rule_date_format(fields)
    warnings += rule_date_order(fields)
    warnings += rule_cnp_length(fields)
    warnings += rule_invalid_characters(fields)
    warnings += rule_low_confidence(fields, ocr_confidence)

    return {
        "missing_fields": rule_missing_fields(fields),
        "warnings": warnings,
        "ocr_confidence": round(ocr_confidence, 2) if ocr_confidence is not None else None,
    }