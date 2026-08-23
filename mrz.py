"""
Supported formats:
    TD3  passports        2 lines x 44 characters
    TD2  ID cards         2 lines x 36 characters
    TD1  ID cards         3 lines x 30 characters
"""
import re

# characters that are legal in an MRZ line
MRZ_CHARSET = re.compile(r'^[A-Z0-9<]+$')

# (number of lines, nominal line length) -> format name
FORMATS = {
    (2, 44): "TD3",
    (2, 36): "TD2",
    (3, 30): "TD1",
}

LENGTH_TOLERANCE = 4

# weights cycle 7-3-1 across the characters being checked
CHECK_WEIGHTS = (7, 3, 1)


def char_value(char):
    if char.isdigit():
        return int(char)
    if char == "<":
        return 0
    return ord(char) - ord("A") + 10


def check_digit(field):
    # compute the ICAO check digit for a field
    total = sum(char_value(c) * CHECK_WEIGHTS[i % 3] for i, c in enumerate(field))
    return str(total % 10)


def verify(field, expected):
    if expected in ("", "<"):
        return None            # no check digit present, nothing to verify
    return check_digit(field) == expected


def parse_date(yymmdd, kind="birth", current_year=2026):
    # MRZ dates are YYMMDD with no century, so the century has to be inferred
    if not re.fullmatch(r'\d{6}', yymmdd):
        return None
    year, month, day = int(yymmdd[:2]), yymmdd[2:4], yymmdd[4:6]
    yy_now = current_year % 100

    if kind == "birth":
        century = 1900 if year > yy_now else 2000
    else:                                   # expiry and similar forward dates
        century = 2000 if year < yy_now + 50 else 1900

    return f"{day}.{month}.{century + year}"


def parse_names(name_field):
    parts = name_field.split("<<")
    surname = parts[0].replace("<", " ").strip()
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    return surname, given


def find_mrz_lines(texts):
    # pick the MRZ lines out of a list of OCR text fragments
    candidates = []
    for text in texts:
        cleaned = text.replace(" ", "").upper()
        if len(cleaned) >= 28 and MRZ_CHARSET.match(cleaned):
            candidates.append(cleaned)
    return candidates


def match_format(lines):
    # find the format whose line length is closest to what OCR produced
    if not lines:
        return None, None
    longest = max(len(line) for line in lines)
    best = None
    for (line_count, nominal), name in FORMATS.items():
        if line_count != len(lines):
            continue
        distance = abs(longest - nominal)
        if distance <= LENGTH_TOLERANCE and (best is None or distance < best[0]):
            best = (distance, name, nominal)
    return (best[1], best[2]) if best else (None, None)


def select_lines(candidates):
    if len(candidates) <= 1:
        return candidates

    ordered = sorted(candidates, key=len, reverse=True)

    # try the plausible groupings, best first
    for size in (2, 3):
        for start in range(len(ordered) - size + 1):
            group = ordered[start:start + size]
            parsed = _parse_exact(group)
            if parsed and any(v is True for v in parsed["checks"].values()):
                # restore the order they appeared in, line order carries meaning
                return sorted(group, key=lambda l: candidates.index(l))
    return candidates


def parse(lines):
    # parse MRZ lines into fields, selecting the real MRZ from OCR noise
    if not lines:
        return None
    return _parse_exact(select_lines([l.strip().upper() for l in lines if l.strip()]))


def _parse_exact(lines):
    # parse an exact set of MRZ lines. Returns None if the format is unknown
    lines = [l.strip().upper() for l in lines if l.strip()]
    if not lines:
        return None

    fmt, nominal = match_format(lines)
    if fmt is None:
        return None

    # pad short lines back to nominal length so the fixed offsets still line up
    lines = [line.ljust(nominal, "<")[:nominal] for line in lines]

    result = {"format": fmt, "checks": {}}

    if fmt in ("TD2", "TD3"):
        line1, line2 = lines
        result["document_type"] = line1[0:2].replace("<", "")
        result["issuing_country"] = line1[2:5].replace("<", "")
        surname, given = parse_names(line1[5:])
        result["family_name"] = surname
        result["given_name"] = given

        result["document_number"] = line2[0:9].replace("<", "")
        result["checks"]["document_number"] = verify(line2[0:9], line2[9])
        result["nationality"] = line2[10:13].replace("<", "")
        result["birth_date"] = parse_date(line2[13:19], "birth")
        result["checks"]["birth_date"] = verify(line2[13:19], line2[19])
        result["gender"] = line2[20].replace("<", "")
        result["expiry_date"] = parse_date(line2[21:27], "expiry")
        result["checks"]["expiry_date"] = verify(line2[21:27], line2[27])

    elif fmt == "TD1":
        line1, line2, line3 = lines
        result["document_type"] = line1[0:2].replace("<", "")
        result["issuing_country"] = line1[2:5].replace("<", "")
        result["document_number"] = line1[5:14].replace("<", "")
        result["checks"]["document_number"] = verify(line1[5:14], line1[14])

        result["birth_date"] = parse_date(line2[0:6], "birth")
        result["checks"]["birth_date"] = verify(line2[0:6], line2[6])
        result["gender"] = line2[7].replace("<", "")
        result["expiry_date"] = parse_date(line2[8:14], "expiry")
        result["checks"]["expiry_date"] = verify(line2[8:14], line2[14])
        result["nationality"] = line2[15:18].replace("<", "")

        surname, given = parse_names(line3)
        result["family_name"] = surname
        result["given_name"] = given

    return result


def to_fields(mrz_data):
    # map parsed MRZ data onto the project's JSON field names
    if not mrz_data:
        return {}
    return {
        "last_name": mrz_data.get("family_name") or None,
        "first_name": mrz_data.get("given_name") or None,
        "date_of_birth": mrz_data.get("birth_date"),
        "date_of_expiry": mrz_data.get("expiry_date"),
        "document_number": mrz_data.get("document_number") or None,
    }
