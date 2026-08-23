# Week 5 - Comparison and evaluation

This week addresses four points raised in review of the week 3 and 4 results:
the test set was not chosen independently, "missing" was conflating two
different failures, the MRZ figure validated the parser rather than the
pipeline, and the difference in abstention behaviour was described but never
quantified.

## 1. A fixed, independently chosen test set

Earlier comparisons used whichever images came first in the folder, or images
selected by their IoU score. The first is an accident, the first three files
turned out to be two card backs and one front where detection had failed. The
second is worse: it selects on the outcome of the method being measured.

`build_testset.py` draws a stratified random sample instead, balanced across
document class and across front/back, with a fixed seed. Selection happens
before either pipeline runs and does not look at any result.

The resulting set is 20 images: 7 ID card backs, 7 ID card fronts, 6 passport
fronts. There is no passport-back bucket because DocXPand passports only have
the data page.

## 2. Four outcome categories

"Missing" was previously doing two jobs. It now splits:

| outcome | meaning |
|---------|---------|
| correct | the value matches ground truth |
| wrong | a value was produced and it does not match |
| missing | the document was localised, but no value was found for this field |
| reject | the document was never localised, so extraction never ran |

The distinction is operational. A reject is a detection failure and the document
can be re-photographed; a missing field is an extraction gap on a document that
was read fine. Only "wrong" is silently damaging.

## 3. Results - classic pipeline

Over the 20-image test set, 104 fields:

| outcome | count | share |
|---------|-------|-------|
| correct | 27 | 26% |
| wrong | 4 | 4% |
| missing | 56 | 54% |
| reject | 17 | 16% |

Three derived figures matter more than the raw accuracy:

- **error rate 4%** - how often it states something untrue
- **abstention 70%** - how often it declines rather than guessing
- **precision when it answers 87%** (27 of 31) - when it does commit to a value,
  it is right almost nine times in ten

This is the quantification of the behaviour described qualitatively in week 4.
The pipeline is not accurate so much as *cautious*: it answers rarely and is
usually right when it does.

### Per field

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

The fields split cleanly in two, and the line is not about difficulty; it is
about whether the field has a machine-recognisable shape.

**Dates work.** 27 of the 31 correct values are dates. A date has a fixed
pattern (`dd.mm.yyyy`) that a regex matches wherever it sits on the page, in any
layout, in any language.

**Names, places, authorities and document numbers do not at all.** Zero
correct across four fields and 20 documents. These have no distinctive shape;
identifying them requires knowing where they sit on that particular layout. The
positional rule written in week 2 was calibrated on the Albanian ID and does not
transfer to DocXPand's French-style layouts, so it never fires.

This is the central finding about rule-based extraction: **it works exactly as
far as regular expressions reach, and stops dead at the boundary.** Adding more
documents does not degrade it gradually; fields either have a pattern or they do
not.

## 4. Detection, and a counter-intuitive result

| side | detected |
|------|----------|
| front | 11 / 13 |
| back | 1 / 7 |

Backs carry fewer high-contrast graphic elements, no photo, no framing, so
edge detection has much less to work with. This matches the week 3 figures (71%
vs 32%) on a properly drawn sample.

But the results matrix shows something that contradicts the assumption the
pipeline is built on. Compare two documents:

- `074f9932` - detection **failed**, yet four fields were extracted correctly,
  including the birth date and both other dates. OCR read the whole passport
  cleanly, MRZ included.
- `1e016d0a` - detection **succeeded** via the rotated-rect fallback, and OCR
  returned nothing usable: fragments like `"0"`, `"9"`, `"[T"`, `"8"`.

When detection fails, `variant_warped` passes the original image through
untouched, and OCR often does better on it. When the fallback succeeds, it crops
approximately (mean IoU 0.595 in week 3) and the crop can clip text or warp it.

**On this dataset the geometric stage is, on balance, harmful to extraction.**
That is specific to DocXPand: the documents are photographed close to frontally,
so there is little perspective to correct and little to gain, while an imprecise
crop actively loses information. On MIDV video frames, where the document is
small and genuinely skewed, the same stage is what makes OCR possible at all.

The lesson is not that the scanner is useless but that its value depends on the
input distribution and has to be measured per dataset rather than assumed.

## 5. MRZ, measured end to end

The week 3 figure of 6498/6498 validated the *parser*: ground-truth MRZ strings
were parsed and compared against ground-truth fields. It showed the ICAO offsets
and check-digit arithmetic were right, and nothing about whether the zone
survives OCR.

Measured properly, locating and reading the zone from the image:

| stage | result |
|-------|--------|
| documents with an MRZ | 13 |
| candidate lines found by OCR | 7 / 13 |
| parsed successfully | 3 / 13 |
| check digits passed on those | 7 / 9 |

**The parser is not the bottleneck; OCR reading the zone is.** Two problems
surfaced while measuring this, both now fixed in `mrz.py`:

- **Line length was matched exactly.** OCR routinely drops the trailing `<`
  filler characters, so a nominal 44-character TD3 line arrives as 42 and was
  rejected outright. Lines are now matched to the nearest known format within a
  tolerance and padded back before the fixed offsets are read.
- **Too many candidates.** Other uppercase strings on the document pass the
  character filter, so one passport produced eight candidate "MRZ lines" of
  which two were real. Groupings are now tried longest-first and accepted only
  if the check digits agree; a wrong grouping essentially never satisfies them,
  which makes the check digits do double duty as a selection criterion.

These fixes took MRZ parsing from 1 of 9 documents to 3 of 9 on the same OCR
output. The remainder fail because OCR fragments the zone into pieces too short
to reconstruct, on one passport the second line came back as 31 characters
instead of 44.

## 6. Comparison with the pretrained pipeline

From week 4, both pipelines on the same images:

| | classic | Donut (DocVQA) |
|---|---------|----------------|
| accuracy | 30% | 10% |
| error rate | low | 90% of answers |
| abstention | 70% | 0% |
| time per document | 7 s | 181 s |

The two fail in opposite ways. The classic pipeline declines seven times out of
ten and is right 87% of the time it commits. Donut never declines, it has no
mechanism to, and was wrong on 18 of 20 fields. Its dominant failure is
misalignment rather than blindness: asked for a surname it returned the MRZ
string, asked for a birth date it returned the surname. It reads the document
and misfiles what it read.

Donut also returns no confidence score, so its answers cannot be filtered by
certainty the way OCR results can.

## 7. Which is more suitable for production

Neither, as they stand, but they are not equally far away, and the distance is
of a different kind in each case.

**The classic pipeline is the safer starting point.** Its 4% error rate and 87%
precision-when-answering mean its output can be trusted where it exists. Its
failures are visible: a missing field is a gap someone can fill, and a reject is
a document that can be re-photographed. Its weaknesses are also specific and
addressable, detection on photographic backgrounds, and the absence of rules
for fields without a regex-friendly shape.

**The pretrained pipeline is not usable without fine-tuning.** A 90% error rate
with no abstention and no confidence score is worse than no answer: it produces
plausible-looking values that a downstream consumer has no way to reject. The
underlying capability is there, it reads text correctly, but nothing connects
that reading to the right field.

For a production system reading identity documents, the deciding factor is not
accuracy but **whether the system knows when it does not know.** The classic
pipeline does; the pretrained model, used off the shelf, does not.

A realistic architecture would use both: rules and MRZ for the fields that have
reliable structure (dates, check-digit-verified numbers), and a model
fine-tuned on identity documents for the fields that need layout understanding
(names, places, authorities), with any model output gated by a confidence
mechanism the current one lacks.

## 8. LayoutLM

Not integrated. Conceptually it sits between the two approaches: it consumes OCR
output together with the position of each text box, so it can learn that "the
string above the nationality line is the surname", the association the rule
based pipeline hard-codes and Donut fails to make. On this evidence it is the
more promising direction of the two model-based options, since the failure
observed with Donut is specifically one of layout association rather than
reading. Integration was not attempted within the available time.

## 9. Limitations

- **20 documents, 2 of 9 DocXPand classes.** The patterns are consistent but the
  absolute numbers are not statistically solid.
- **Images are capped at 2 megapixels before OCR**, a constraint imposed by
  running on a machine with 8 GB of RAM. Full-resolution OCR might read the MRZ
  more reliably; the reported MRZ figures are a lower bound.
- **Extraction was run one process per image** for the same reason. This does
  not affect results, only how they were produced.
- **Donut was compared on 4 documents, not the full 20** - 181 seconds per
  document on CPU makes the full set impractical.
- **`issued_by` is rejected 8 times of 13**, meaning most of its documents were
  never localised. Its zero score reflects detection more than extraction.