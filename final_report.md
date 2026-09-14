# Document Scanner & Verifier - Final Report

A comparison of two approaches to reading identity documents: a classic
computer-vision pipeline built from OpenCV, OCR and hand-written rules, and a
pretrained document-understanding model used off the shelf. Both produce the
same JSON schema and were evaluated against the same ground truth, so the
comparison rests on measurement rather than impression.

## 1. The problem

Extracting structured data from a photograph of an identity document breaks into
three stages, each of which can fail independently:

1. **Localisation** - find the document in the scene and correct its perspective
2. **Recognition** - read the text off it
3. **Structuring** - decide which text belongs in which field

The classic pipeline addresses these in sequence with explicit code. The
pretrained model collapses all three into one forward pass. Each design has a
characteristic failure mode, and those turned out to matter more than the
accuracy difference between them.

## 2. Datasets, and why they changed

**MIDV-500** was used first: video frames of identity document specimens under
varied capture conditions. Measurement showed the limit quickly. After
perspective correction the document came out at roughly 560×900 pixels with
glyphs of 15–20 px, and OCR read "ROMANIA" as "HOMANA", with most fields below
0.4 confidence.

Sharpness measured as the variance of the Laplacian puts the problem in one
number:

| source | size | sharpness |
|--------|------|-----------|
| MIDV video frame, perspective-corrected | 960×583 | 15 |
| MIDV document template (clean scan) | 624×391 | 1475 |
| DocXPand scene, perspective-corrected | ~1400×900 | 790 |

The frames are not short of pixels, they are short of detail, because of motion
blur, video compression and interpolation through the warp.

Seven alternatives were evaluated:

| dataset | size | verdict |
|---------|------|---------|
| DocXPand-25k | 17 GB | **chosen** - real backgrounds, corner *and* text ground truth |
| MIDV-2020 | 124 GB | requires a request form and sFTP access |
| IDNet | 46 GB | too large, oriented towards fraud detection |
| Kaggle passport-dataset | 41 MB | 11 images, no text annotations |
| generated-passports-segmentation | 58 MB | document fills the frame, nothing to detect |
| MRZ detection set | 1 MB | cropped MRZ strips only |
| Vietnamese passports | 27 MB | noisy grayscale scans |

`DocXPand-25k` was the only one annotating **both** the document corners and every
printed field. Without that, none of the required metrics are computable.

The 17 GB is never stored: `download_docxpand.py` streams the archive parts
straight into `tar` and writes only the document images, keeping disk usage in
the hundreds of MB.

MIDV was retained as the robustness set, its hard capture conditions contain
degradation a rendered dataset does not reproduce.

## 3. Pipeline 1 - classic

`image → detect → warp → enhance → OCR → rules → validate → JSON`

### Localisation

The initial detector looked for the largest contour simplifying to exactly four
corners. On MIDV's hard conditions it found the document in 2 of 14 images.

Four modifications were tested independently:

| variant | detection rate |
|---------|----------------|
| baseline | 2 / 14 |
| relaxed area threshold | 2 / 14 |
| adaptive Canny thresholds | 2 / 14 |
| morphological closing | 2 / 14 |
| rotated-rectangle fallback | 7 / 14 |

Only the fallback moves the number, and this is the substantive result: the
other three change *how* edges are found, but on these images a closed
four-corner contour does not exist at all, clutter and fingers interrupt the
border. Changing the shape model works where tuning thresholds cannot.

**A naive fallback inflates the metric.** Combining it with looser thresholds
gave 12/14, until the outputs were inspected: several "detections" had aspect
ratios of 0.56 and sizes of 1075×1916, the whole frame. Two geometric gates
were added: the quad must cover 6–90% of the frame, and its aspect ratio must
fall within 0.45 of the ID-1 standard (85.6/54 = 1.585).

Final result on MIDV hard conditions: **2/14 → 11/14**, at 3 ms per image versus
5 ms for the baseline, the fallback only runs when the first pass fails.

One false positive survives: on `PS38_01` the detector locked onto a map in the
background whose proportions fall inside the accepted range. The gates are purely
geometric; rejecting that would require content awareness.

### MRZ

The machine-readable zone is the one part of a document parseable by rule alone:
ICAO 9303 fixes every value at a known character offset, and check digits allow
a reader to verify what it read. `mrz.py` implements TD1, TD2 and TD3 with check
digit verification.

Two bugs surfaced only when the stage was measured end to end, and both are
instructive:

- **Exact line-length matching.** OCR routinely drops the trailing `<` filler,
  so a nominal 44-character TD3 line arrives as 42 and was rejected outright.
- **Too many candidates.** Other uppercase strings pass the character filter,
  one passport produced eight candidate "MRZ lines" of which two were real.

Both are fixed: lines are matched to the nearest format within a tolerance and
padded back, and groupings are accepted only if their check digits agree. A
wrong grouping essentially never satisfies the check digits, which makes them do
double duty as a selection criterion.

### Validation

Rules report problems and never modify data, keeping the output
self-describing: missing fields, invalid dates, date ordering, CNP length,
invalid characters in name fields, low OCR confidence, MRZ check digits, and
disagreements between the MRZ and the printed text.

The date-range rule caught a real OCR error during evaluation: a birth year read
as 1888 instead of 1989.

## 4. Pipeline 2 - pretrained model

`image → model → JSON`

**`donut-base`, the plain pretrained backbone, is unusable here.** On an identity
document the decoder degenerates, emitting one token repeatedly:
`I D D D D D D D...`. A pretrained backbone has no task; without fine-tuning on
something it has no output format to produce.

**`donut-base-finetuned-docvqa`** was used instead. Rather than requesting a
structured record, each field is asked as a question ("What is the date of
birth?"), which suits identity documents well.

Per the brief, no fine-tuning was performed. Each question is a separate forward
pass: roughly 25 seconds on CPU, so a full document takes about three minutes.

## 5. Evaluation methodology

Three methodological problems were corrected during the project, and each
changed the conclusions.

**The test set was not chosen independently.** Early comparisons used whichever
files came first, which happened to be two card backs and one front where
detection had failed, or files selected by their IoU score, which selects on
the outcome being measured. The final evaluation uses a stratified random sample
balanced across document class and side, with a fixed seed, drawn before either
pipeline ran.

**"Missing" was doing two jobs.** It now splits into *missing* (the document was
localised but no value was found) and *reject* (the document was never
localised, so extraction never ran). A reject is a detection failure and the
document can be re-photographed; a missing field is an extraction gap on a
document that was read fine.

**The MRZ figure measured the wrong thing.** An earlier result of 6498/6498
validated the parser: ground-truth MRZ strings were parsed and compared against
ground-truth fields. It showed the ICAO offsets were right and nothing about
whether the zone survives OCR.

Field comparison normalises case, accents and punctuation, since "MARECHAL
DUBOIS" for "Maréchal-Dubois" is a formatting difference, not an extraction
error.

## 6. Results

### Classic pipeline, fixed test set (20 documents, 104 fields)

| outcome | count | share |
|---------|-------|-------|
| correct | 27 | 26% |
| wrong | 4 | 4% |
| missing | 56 | 54% |
| reject | 17 | 16% |

Three derived figures say more than the accuracy:

- **error rate 4%** - how often it states something untrue
- **abstention 70%** - how often it declines rather than guessing
- **precision when it answers 87%** (27 of 31)

The pipeline is not so much accurate as **cautious**: it answers rarely and is
usually right when it does.

### Per field - a sharp boundary

| field | correct | wrong | missing | reject |
|-------|---------|-------|---------|--------|
| date_of_expiry | 11 | 1 | 0 | 1 |
| date_of_issue | 10 | 0 | 1 | 2 |
| date_of_birth | 6 | 1 | 6 | 0 |
| last_name | 0 | 1 | 11 | 1 |
| first_name | 0 | 0 | 11 | 2 |
| place_of_birth | 0 | 0 | 11 | 2 |
| document_number | 0 | 1 | 11 | 1 |
| issued_by | 0 | 0 | 5 | 8 |

**27 of the 31 correct values are dates.** Names, places, authorities and
document numbers score zero across all 20 documents.

The dividing line is not difficulty, it is whether the field has a
machine-recognisable *shape*. A date matches `dd.mm.yyyy` wherever it sits, in
any layout, in any language. A surname has no distinctive form; identifying it
requires knowing where it sits on that particular document. The positional rule
written for the Albanian ID does not transfer to DocXPand's French-style
layouts, so it never fires.

**Rule-based extraction works exactly as far as regular expressions reach, and
stops dead at the boundary.** It does not degrade gradually across documents;
fields either have a pattern or they do not.

### Detection

| side | detected |
|------|----------|
| front | 11 / 13 |
| back | 1 / 7 |

Backs carry no photo and no framing, so edge detection has far less to work
with.

### A counter-intuitive finding

Two documents from the same test set:

- `074f9932` - detection **failed**, yet four fields were extracted correctly
  including all three dates. OCR read the whole passport cleanly, MRZ included.
- `1e016d0a` - detection **succeeded** via the fallback, and OCR returned
  fragments: `"0"`, `"9"`, `"[T"`, `"8"`.

When detection fails the pipeline passes the original image through untouched,
and OCR often does better on it. When the fallback succeeds it crops
approximately (mean IoU 0.595) and can clip or warp text.

**On DocXPand the geometric stage is, on balance, harmful to extraction.** The
documents are photographed close to frontally, so there is little perspective to
correct and much to lose from an imprecise crop. On MIDV video frames, where the
document is small and genuinely skewed, the same stage is what makes OCR
possible at all.

The value of preprocessing depends on the input distribution and must be
measured per dataset rather than assumed.

### MRZ, end to end

| stage | result |
|-------|--------|
| documents with an MRZ | 13 |
| candidate lines found by OCR | 7 / 13 |
| parsed successfully | 3 / 13 |
| check digits passed on those | 7 / 9 |

The parser is not the bottleneck; OCR reading the zone is. Where the zone is
read, the check digits confirm it, but the zone survives OCR only about a
quarter of the time. On one passport the second line came back as 31 characters
instead of 44.

### Classic versus pretrained

The week 4 comparison used four images chosen by detection quality, while the
classic pipeline was evaluated on the seeded stratified test set; two different
samples, so the two were never scored on identical input.
`run_donut_testset.py` closes that gap: it runs Donut over the same
`testset.json`, one process per image, and the results below cover **all 20
documents** with both pipelines asked for the same fields and scored against the
same ground truth.

| | classic | Donut (DocVQA) |
|---|---------|----------------|
| correct | 27 | 18 |
| wrong | **4** | **86** |
| missing | 56 | 0 |
| reject | 17 | 0 |
| accuracy | 26% | 17% |
| abstention | **70%** | **0%** |
| **precision when it answers** | **87%** (27/31) | **17%** (18/104) |
| time per document | 7 s | 181 s |
| confidence score | yes (per OCR region) | none |

On the full set the accuracy gap narrows to three to two, but the precision gap
does not move: **87% against 17%.** In absolute terms the classic pipeline made
four false statements across 20 documents; Donut made **86**.

They fail in opposite ways. The classic pipeline has explicit paths for "no rule
matched" and "no document was localised", so it declines on 70% of fields and is
right on 87% of the ones it commits to. Donut has no mechanism for declining,
every question produces an answer whether or not the information is on the page,
and it returns no confidence score, so nothing downstream can filter them.

Its dominant failure is **misalignment rather than blindness**:

| field | ground truth | Donut answered |
|-------|--------------|----------------|
| document_number | XQ6D4PW94 | `xq6d4pw94` - correct |
| last_name | Maréchal-Dubois | `xq6d4pw948pil5011287` - the MRZ string |
| date_of_birth | 28.11.1950 | `marchal-dubois` - the surname |

On another document, asked for a given name it answered `kaster`; a garbled
version of the surname, which the classic pipeline read correctly as `KOSTER`.
The information is seen, misfiled, and degraded.

Two further patterns: the same string returned for several different questions
(`ministere des affairs` for surname, given name, authority and document
number), and occasional invented text (`margarine` as a surname).

**The model reads the document but does not understand what is being asked of
it.** The reading is already there, so fine-tuning on identity documents would
likely fix the alignment.

## 7. Conclusions: which is more suitable for production

Neither, as they stand. But they are not equally far away, and the distance is
of a different kind.

**The classic pipeline is the safer starting point.** A 4% error rate and 87%
precision-when-answering mean its output can be trusted where it exists. Its
failures are visible and actionable: a missing field is a gap someone can fill,
a reject is a document that can be re-photographed. Its weaknesses are specific
and addressable, detection on photographic backgrounds, and the absence of
rules for fields without a regex-friendly shape.

**The pretrained pipeline is not usable off the shelf.** A 90% error rate with
no abstention and no confidence score is worse than no answer: it produces
plausible-looking values a downstream consumer has no way to reject.

For a production system reading identity documents, the deciding factor is not
accuracy but **whether the system knows when it does not know.** The classic
pipeline does; the pretrained model, used without fine-tuning, does not.

A realistic architecture would use both: rules and MRZ for fields with reliable
structure, dates, and numbers verifiable by check digit, and a model
fine-tuned on identity documents for fields needing layout understanding, with
its output gated by a confidence mechanism the current one lacks.

## 8. Limitations

- **20 documents, 2 of 9 DocXPand classes.** Patterns are consistent, absolute
  numbers are not statistically solid.
- **Images capped at 2 megapixels before OCR**, forced by running on 8 GB of
  RAM. Full-resolution OCR might read the MRZ more reliably, so those figures
  are a lower bound.
- **Donut now covers the full 20-document test set.** At 181 s per document this takes about an hour on CPU; it is run one process per image because loading Donut and EasyOCR repeatedly in a single process exhausts 8 GB of memory.
- **No fine-tuning**, per the brief. The comparison is therefore between
  hand-written rules and an off-the-shelf model, not between rules and what a
  trained model could do.
- **LayoutLM not integrated.** Conceptually it sits between the two: it consumes
  OCR output *with* the position of each text box, so it can learn the
  association the rules hard-code and Donut fails to make. Given that Donut's
  failure is specifically one of layout association rather than reading, this is
  the more promising of the two model-based directions.

## 9. Possible next steps

- Improve detection on photographic backgrounds, currently the weakest stage,
  and exactly where a learned model should help.
- Measure the impact of binarisation on OCR accuracy; the variants are
  implemented but only partially compared.
- Combine results across the several frames of a MIDV clip, which show the same
  document, by majority vote or best confidence.
- Fine-tune a document-understanding model on identity documents and re-run the
  same evaluation, which the fixed test set now makes directly comparable.
