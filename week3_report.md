# Week 3 — Robustness, MRZ and Measurement

## Why the dataset changed

Week 2 ran on MIDV-500 video frames. Measurement showed the limit clearly: after
perspective correction the document came out around 560×900 px with glyphs of
15–20 px, and OCR read "ROMANIA" as "HOMANA" with most fields below 0.4
confidence. A sharpness comparison (variance of the Laplacian) puts the problem
in one number:

| source | size | sharpness |
|--------|------|-----------|
| MIDV video frame, after perspective correction | 960×583 | 15 |
| MIDV document template (clean scan) | 624×391 | 1475 |
| DocXPand scene, after perspective correction | ~1400×900 | 790 |

The MIDV frames are not short of pixels, they are short of detail, because of
motion blur, video compression and interpolation through the warp.

Seven alternatives were evaluated before choosing:

| dataset | size | verdict |
|---------|------|---------|
| DocXPand-25k | 17 GB | **chosen** — real backgrounds, corner + text ground truth |
| MIDV-2020 | 124 GB | requires a request form and sFTP access |
| IDNet | 46 GB | too large, oriented towards fraud detection |
| Kaggle passport-dataset | 41 MB | only 11 images, no text annotations |
| generated-passports-segmentation | 58 MB | document fills the frame, nothing to detect |
| MRZ detection set | 1 MB | cropped MRZ strips only, no full documents |
| Vietnamese passports | 27 MB | noisy grayscale scans |

DocXPand-25k was selected because it is the only one that annotates **both** the
document corners and every printed field. Without that, none of the metrics this
week asks for can be computed.

The 17 GB is not stored: the archive parts are streamed straight into `tar` and
only the document images are written to disk. `download_docxpand.py` does this
and keeps a bounded number of images per class, so disk usage stays in the tens
of MB.

MIDV-500 is kept as the robustness set, its hard capture conditions (cluttered
background, held in hand, partially out of frame) contain degradation that a
rendered dataset does not reproduce.

## 1. Improved document detection

The week 1 detector looks for the largest contour that simplifies to exactly
four corners and covers more than 15% of the frame. On MIDV hard conditions it
found the document in 2 of 14 images.

Four modifications were tested separately:

| variant | change | detection rate |
|---------|--------|----------------|
| baseline | as in week 1 | 2 / 14 |
| relaxed area | threshold 15% → 8% | 2 / 14 |
| adaptive Canny | thresholds from image median | 2 / 14 |
| morphological closing | 9×9 closing to join broken edges | 2 / 14 |
| rotated-rect fallback | `cv2.minAreaRect` when no clean quad exists | 7 / 14 |

Only the fallback moves the number. The other three change *how* edges are
found, but on these images a closed four-corner contour does not exist at all,
clutter and fingers interrupt the border. Changing the shape model works where
tuning thresholds cannot.

### A naive fallback inflates the metric

Combining the fallback with relaxed thresholds gave 12/14, which looked like a
win until the straightened outputs were inspected: several had aspect ratios of
0.56 and sizes of 1075×1916, the detector had locked onto the image border, not
the document. Two geometric gates were added:

- **area** - the quad must cover between 6% and 90% of the frame
- **aspect ratio** - a straightened ID-1 card has a ratio of 85.6/54 = 1.585;
  detections further than 0.45 from that (in either orientation) are rejected

Final result on MIDV hard conditions: **2/14 → 11/14**, at 3 ms per image versus
5 ms for the baseline. The fallback only runs when the first pass fails.

Visual inspection of the 11 detections found 8 correct, 2 partial (the card is
found but cropped), and **1 false positive**, on `PS38_01` the detector locked
onto a map in the background whose proportions fall inside the accepted range.
The gates are purely geometric, so any rectangular object of card-like shape can
pass. Rejecting it would need content awareness.

## 2. Measured on DocXPand with ground truth

DocXPand annotates the document corners, so IoU can be computed directly instead
of judged by eye. Measured over the full local subset of 4555 images:

| metric | value |
|--------|-------|
| documents detected | 2540 / 4555 (56%) |
| mean IoU (when detected) | 0.619 |
| median IoU | 0.683 |
| IoU >= 0.5 | 1578 / 2540 (62%) |
| mean detection time | 2 ms |

Per class:

| class | detected |
|-------|----------|
| ID_CARD_TD2_A | 1976 / 3575 (55%) |
| PP_TD3_A | 564 / 980 (58%) |

Three patterns come out of the per-image results.

**The front of a document is detected more than twice as often as the back**
(71% vs 32%). The front carries a photo, framing lines and graphic elements that
produce strong edges; the back is largely uniform, so Canny has far less to work
with. This is a property of the documents, not of the parameters, no threshold
change recovers an edge that is not there.

**The fallback does most of the work, at lower precision.** Of the 2540
detections, 2058 came from the rotated-rectangle fallback and only 482 from a
clean four-corner contour. Their quality differs accordingly:

| method | detections | mean IoU |
|--------|-----------|----------|
| clean contour | 482 | 0.723 |
| rotated-rect fallback | 2058 | 0.595 |

The fallback is what lifts the detection rate, but it fits a rectangle around
the largest plausible contour rather than following the document border, so its
localisation is looser. This is a measured trade-off rather than a free gain.

**The IoU distribution is polarised**, not gradual:

| IoU band | share |
|----------|-------|
| >= 0.9 | 16% |
| 0.7 – 0.9 | 31% |
| 0.5 – 0.7 | 15% |
| 0.3 – 0.5 | 14% |
| < 0.3 | 24% |

Almost half the detections (47%) are at IoU >= 0.7 and usable downstream, while
a quarter are below 0.3 and effectively wrong. The detector either locks onto
the document or latches onto something else, it rarely degrades gently. For a
pipeline that feeds OCR, that is arguably the better failure mode, because a
badly localised crop is easier to reject than a subtly skewed one.

Inspecting the ground-truth areas explains part of the remaining failures: some
documents cover more than 100% of the frame, they extend past the image border,
and are correctly rejected by the area gate. Others fail at 33–49% coverage,
where the gates are not the constraint: DocXPand backgrounds are photographs of
real surfaces, so Canny produces many competing contours and the document border
is not the strongest one.

## 2b. Field extraction, measured

Run over 20 images with OCR (10.4 s per document on CPU):

| outcome | count |
|---------|-------|
| correct | 25 / 80 (31%) |
| wrong | 3 / 80 (4%) |
| missing | 52 / 80 (65%) |

| field | correct | wrong | missing |
|-------|---------|-------|---------|
| date_of_expiry | 6 | 0 | 4 |
| date_of_issue | 6 | 0 | 4 |
| date_of_birth | 4 | 0 | 6 |
| last_name | 4 | 0 | 6 |
| first_name | 3 | 1 | 6 |
| document_number | 2 | 2 | 6 |
| issued_by | 0 | 0 | 10 |
| place_of_birth | 0 | 0 | 10 |

Three things stand out.

**The error rate is low: 3 wrong out of 80.** When the pipeline reports a value
it is almost always right. For a document reader that is the preferable failure
mode, a missing field is visible and can be re-checked, a silently wrong one is
not.

**Two fields are never extracted.** `issued_by` and `place_of_birth` are 0/10
each, and this is structural rather than accidental: neither appears in the MRZ
(ICAO 9303 has no field for issuing authority or place of birth), and no
positional rule was written for them. They are known gaps, not failures.

**Most of the extracted values come from the MRZ, not from the printed text.**
On a representative document the positional rules found a single date, while the
MRZ supplied name, document number, birth date and expiry, all four verified by
check digits. Compared against ground truth:

| field | ground truth | extracted |
|-------|--------------|-----------|
| last_name | De Sousa | DE SOUSA |
| first_name | René, Adrien, Jules | RENE ADRIEN JULES |
| date_of_birth | 29.12.1934 | 29.12.1934 |
| document_number | EL7U9ZC85 | EL7U9ZC85 |
| place_of_birth | Delahaye (69) | — |

The name differences are formatting only: the MRZ is uppercase and unaccented by
standard. This is the practical case for handling the MRZ separately, fixed
offsets and check digits beat positional guessing.

## 2c. Preprocessing variants compared against ground truth

`compare_variants.py` counts text regions, which measures OCR output but not
correctness. Comparing variants against ground-truth field values instead, on
the images where detection succeeded:

| variant | correct fields |
|---------|----------------|
| raw (no geometric correction) | 7 / 36 (19%) |
| warped (perspective corrected) | 6 / 36 (17%) |
| warped + 2x upscale | no change |

**Perspective correction does not improve extraction on this dataset, and
slightly hurts it.** That is contrary to the expectation the pipeline was built
on, and the explanation is specific to DocXPand: the documents are photographed
close to frontally, so there is little perspective to correct, while the
rotated-rect fallback localises loosely (mean IoU 0.595) and its crop sometimes
clips text. The correction introduces more degradation than it removes.

Upscaling changes nothing, which is consistent, interpolation adds no detail
that is not already there, and at a sharpness of 790 these images are not
resolution-limited.

This also explains an otherwise confusing result: over the 20-image run,
undetected images scored *better* than detected ones (43% vs 17% of fields
correct). When detection fails the pipeline falls back to the original frame,
which on this dataset is the better OCR input.

The conclusion is not that the scanner is useless, on MIDV frames, where the
document is small and genuinely skewed, it is what makes OCR possible at all,
but that its value depends on the input distribution, and should be measured per
dataset rather than assumed.

## 3. MRZ

Both DocXPand document families carry a machine-readable zone, so the MRZ branch
of the specification applies here (neither the Romanian licence nor the Albanian
ID has one).

`mrz.py` implements ICAO 9303: detection, parsing for TD1/TD2/TD3, and check
digits. The MRZ is worth handling separately because its layout is fixed,
every value sits at a known offset, so no positional guessing is needed, and the
check digits allow verification of the read.

Validated against DocXPand ground truth on all annotated documents:

| field | agreement with the printed fields |
|-------|-----------------------------------|
| document number | 6498 / 6498 (100%) |
| family name | 434 / 500 (87%) |
| birth date | ~59% |
| expiry date | ~63% |

The document number matches perfectly. The name differences are formatting only
(accents and hyphens: "MARECHAL DUBOIS" vs "Maréchal-Dubois").

The date figures needed investigation, and produced two findings:

- **717 of the mismatches are day/month swaps.** The visual field and the MRZ
  disagree in the dataset itself, so this is a property of DocXPand, not of the
  parser.
- **111 are century ambiguity.** MRZ dates are YYMMDD with no century, so "22"
  is either 1922 or 2022. The parser resolves birth dates to the past and expiry
  dates to the future, which is the standard convention, but the ambiguity is
  inherent. Using a single fixed pivot silently parsed an expiry of "33" as 1933,
  a bug found only because ground truth was available to check against.

Because the MRZ can disagree with the printed text, the pipeline does not
silently overwrite: MRZ values fill fields that are empty, and disagreements are
reported as validation warnings.

## 4. Validation module

Validation moved out of `extract.py` into `validation.py` and was extended:

| rule | what it catches |
|------|-----------------|
| missing fields | fields that could not be extracted |
| date format | day/month/year out of range — caught the OCR error that read birth year 1989 as 1888 |
| date order | birth before issue, issue before expiry |
| CNP length | must be exactly 13 digits |
| invalid characters | digits or noise inside name fields |
| low confidence | flags results the OCR engine was unsure about |
| MRZ check digits | verifies the MRZ was read correctly |
| MRZ vs printed text | reports disagreements instead of resolving them |

Rules report problems and never modify data, so the output stays
self-describing.

## 5. Metrics tooling

- `evaluate.py` = detection rate per MIDV condition, method used, runtime,
  failure list
- `evaluate_docxpand.py` = IoU against ground-truth corners, and per-field
  correct/wrong/missing counts. Field comparison normalises case, accents and
  punctuation, since those differences are not extraction errors
- `compare_variants.py` = runs OCR over each preprocessing variant on the same
  images and reports regions found, confident regions, mean confidence and
  pattern hits

## 6. Limitations

- **Detection does not transfer unchanged across datasets.** 79% on MIDV hard
  conditions, 56% on DocXPand. The gap is not a tuning failure so much as a
  difference in image distribution: DocXPand backgrounds are photographic, so
  edge strength no longer separates the document from its surroundings.
- **Backs of documents are the weak case** (32% vs 71% for fronts), because they
  carry fewer high-contrast graphic elements.
- **False positives remain possible.** The gates are geometric; a rectangular
  object of similar proportions passes.
- **Documents extending past the frame are rejected by design.** The area gate
  cannot distinguish them from whole-frame false positives.
- **Only 2 of the 9 DocXPand classes are covered** (4555 images). Reaching the
  other classes means streaming more archive parts.
- **MRZ century resolution is heuristic** and will misread documents belonging
  to people over ~100 years old.
- **`issued_by` and `place_of_birth` are never extracted**, since neither is in
  the MRZ and no positional rule covers them.
- **Perspective correction is not universally beneficial** - on DocXPand it
  slightly reduces extraction accuracy.
