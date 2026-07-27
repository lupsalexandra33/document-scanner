# OCR Pipeline — Observed Limitations (Week 2)

This document describes the limitations observed while building and testing the classic OCR pipeline (`extract.py`): image → scanner → EasyOCR → field extraction rules → JSON. Tests were run on the MIDV-500 dataset (Romanian driving licence and Albanian ID), on the "clean background" conditions (TA / TS / KS).

## Summary

The pipeline works end to end and produces valid JSON, but the quality of the extracted fields depends heavily on the input image and the document layout. On the Albanian ID it extracts most fields correctly; on the Romanian licence it extracts dates and the document number, but fails on names and the personal number.

## 1. Input resolution is the main bottleneck

MIDV-500 provides video frames (1920×1080) where the document is a small object in the scene. After perspective correction the document comes out at roughly 560×900 pixels, so the printed text is only ~15-20 pixels tall, below what EasyOCR needs for reliable recognition.

This is visible in the confidence scores: on Romanian frames most fields are read with confidence 0.02-0.5, and words come out corrupted ("ROMANIA" read as "HOMANA", "PERMIS" as "Peamis"). The scanner itself works correctly (it detects and straightens the document); the limitation is the source resolution, not the preprocessing.

## 2. Character confusion

EasyOCR frequently confuses visually similar characters, especially on small text:

- digit/digit: `9` read as `8` (birth year 1989 read as 1888 in TA38_25)
- digit/letter: `B` read as `8` (document number `B10724188` read as `810724188`)
- letter in a number: `08` read as `0a`, making the date `10.08.2008` unparseable

Because of the last case, one of the three dates on the Romanian licence is usually lost, the regex correctly rejects `10.0a.2009` since it is not a valid date shape.

## 3. Names are layout- and quality-dependent

Name extraction is position-based and only implemented for the Albanian ID, where the surname sits above the given name in the left column and OCR reads them with confidence 1.00 (e.g. "Sojli", "Monika").

On the Romanian licence the same approach fails: the names are read with confidence 0.18-0.50 and come out corrupted ("TOGWMXIDVIRUK" read as "Tc ixidvauk"), so they are left null. This shows a core weakness of the rule-based approach: extraction
rules are specific to each document layout and do not transfer between documents.

## 4. Fields with identical shape are hard to separate

The document number (9 characters) and parts of the personal number (CNP, 13 digits) have overlapping shapes, so a naive regex matches both. The pipeline uses the OCR confidence score as a tie-breaker (keeping only matches above 0.4), which removes most false positives but is not fully reliable.

The CNP itself is usually too corrupted to extract correctly (read as `890306188351` instead of `1890306158351`), so it is generally left null.

## 5. Date assignment is heuristic

The three dates (birth, issue, expiry) all have the same `dd.mm.yyyy` shape, so they cannot be told apart by pattern alone. The pipeline assigns them by year:

- oldest date → date of birth
- when only two dates are found, the gap between them is used: a small gap is treated as issue + expiry (licences are valid 5-10 years), a large gap as birth + expiry

This works for typical cases but can misassign dates in unusual ones, and depends on OCR reading the years correctly.

## Validation

The pipeline includes basic validation rules that flag problems without fixing them:

- missing fields (null values)
- invalid dates (day/month/year out of range), e.g. it correctly flagged the 1888 birth year caused by OCR error
- CNP length (must be 13 digits)
- lowest OCR confidence among extracted fields, reported per document

These rules make the output self-describing: a consumer can see which fields are missing or suspicious. They do not correct the errors, that would require either a higher-quality image source or a layout-aware model.

## Conclusion

The classic pipeline is transparent and controllable, but fragile: it depends on input quality and requires document-specific rules. The Albanian ID (clean OCR) works well; the Romanian licence (corrupted OCR due to low resolution) works only partially. Both the resolution limit and the per-document rules motivate comparing this approach with a pretrained document-understanding model in the coming weeks.