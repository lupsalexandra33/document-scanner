# Week 4 - Pipeline 2: a pretrained document-understanding model

Pipeline 1 detects the document, runs OCR and applies hand-written rules.
Pipeline 2 hands the image to a pretrained model and reads structured data out
of it, no detection, no regex, no layout rules. Both write the same JSON
schema, so their results can be compared directly.

Per the brief, the model is used as-is: no training and no fine-tuning. What it
produces is recorded as it comes, mistakes included.

## 1. Choosing a variant

**`donut-base`** = the plain pretrained backbone. On an identity document it
degenerates: the decoder emits the same token repeatedly.

```
<s_cord-v2> I D D D D D D D D D D D D D D D D D D D D D D D ...
```

This is not a bug in the integration. A pretrained backbone has no task; without
fine-tuning on something, it has no output format to produce. It was discarded.

**`donut-base-finetuned-docvqa`** = fine-tuned for document visual question
answering, which suits identity documents well: instead of asking for a whole
structured record, each field is requested with a question ("What is the date of
birth?"). This variant answers coherently and is the one used.

`donut-base-finetuned-cord-v2` was not used: it is fine-tuned on receipts, whose
field set (items, prices, totals) does not overlap with identity documents.

## 2. Integration

`donut_pipeline.py` asks one question per field of the shared schema, then
normalises the answers:

- dates come back as `25 01 1954` and are rewritten to `25.01.1954`
- answers such as "unanswerable" or "none" are mapped to `null`
- the raw answers are kept in the output, so a wrong value stays visible rather
  than being quietly cleaned away

Each question is a separate forward pass, so cost scales with the number of
fields: about 25 seconds per question on CPU, roughly 78 seconds for a document
with three fields and around twelve minutes for all eight.

## 3. Comparison on the same images

An initial run used the first three images in the folder, which turned out to be
two card backs and one front where detection failed, an accidental sample
rather than a chosen one. The comparison below instead uses four **fronts with
high detection quality** (IoU between 0.84 and 0.95), selected from the week 3
evaluation results, so that the extraction stage is measured on input it was
actually given correctly.

Both pipelines were asked for the same eight fields and scored against the same
ground truth with the same comparison function (case, accents and punctuation
normalised).

| pipeline | correct | wrong | missing | accuracy | avg time |
|----------|---------|-------|---------|----------|----------|
| classic (OCR + rules) | 6 | 2 | 12 | **30%** | 7 s |
| Donut (DocVQA) | 2 | **18** | 0 | 10% | 181 s |

The classic pipeline is three times more accurate and twenty-six times faster.

The failure behaviour differs more than the accuracy does. Donut was wrong on 18
of 20 fields and left nothing blank, it has no mechanism for declining, so
every question produces an answer whether or not the information is there. The
classic pipeline was wrong twice and silent twelve times.

Per image, the classic pipeline behaves in two distinct regimes:

| image | classic | donut |
|-------|---------|-------|
| KOSTER | 4 correct, 0 wrong | 1 correct, 4 wrong |
| SOTO NUNEZ | 2 correct, 2 wrong | 1 correct, 4 wrong |
| two others | 0 correct, 0 wrong, all missing | 0 correct, 5 wrong |

It either reads the document well or produces nothing at all; it rarely emits
plausible-looking rubbish. Donut is uniformly wrong across all four. For a
document reader the former is the more workable failure mode: a missing field is
visible and can be re-read, a confidently wrong one propagates unnoticed. Donut
also returns no confidence score, so its answers cannot be filtered by certainty
the way OCR results can.

## 4. What the model actually gets wrong

Three failure modes appear in the results.

**Field misalignment, the most common.** The model reads text that really is on
the document, then attaches it to the wrong field:

| field | ground truth | Donut answered |
|-------|--------------|----------------|
| document_number | XQ6D4PW94 | `xq6d4pw94` - correct |
| last_name | Maréchal-Dubois | `xq6d4pw948pil5011287` - the MRZ string |
| date_of_birth | 28.11.1950 | `marchal-dubois` - the surname |

On the KOSTER document the same pattern appears with corruption on top: asked
for the given name, Donut answered `kaster` - a garbled version of the surname,
which the classic pipeline read correctly as `KOSTER`. The information is seen,
misfiled, and degraded.

**Answer repetition.** On one image, `ministere des affairs` was returned for
four different questions: surname, given name, issuing authority and document
number. On another, `26.06.1996` was returned for the birth date, the issue
date, the expiry date and the document number. The model latches onto one
salient string and reuses it.

**Invented text.** Sometimes the answer appears nowhere on the document:
`margarine` and `tobaccoffee` as a surname and an issuing authority,
`cigarvival` in an earlier test.

**The model sees the document but does not understand what is being asked of
it.** That is a more specific diagnosis than "the model is wrong", and it points
somewhere: the reading is already there, so fine-tuning on identity documents
would likely fix the alignment. A model that could not read the text at all
would be a much harder problem.

## 5. Sensitivity to input

Two of the three test images were the back of an ID card, where ground truth
covers only issue date, expiry and authority. Donut answered every question on
them anyway, and every answer was wrong - it returned street names, a URL
fragment (`t.ly/98st`) and the issuing ministry for fields such as date of birth
and place of birth.

The classic pipeline did better on exactly those images: it extracted both dates
correctly from each back, because a date regex does not care which side of the
card it is reading. The front, which carries the MRZ, is where the classic
pipeline depends on detection succeeding - and on this particular front it
failed to detect the document, which is why it scored zero there while Donut got
the document number.

This matches what the classic pipeline shows on the detection side (fronts
detected at 71%, backs at 32%): the back of a document is the harder case for
both approaches, because it carries fewer distinctive visual features.

## 6. Conclusions

- **A pretrained backbone alone is unusable.** `donut-base` produces degenerate
  output on this domain; a task-fine-tuned variant is the minimum.
- **The modern pipeline loses to hand-written rules here** - 9% against 36% -
  which is the opposite of the expected result. Without fine-tuning on identity
  documents, DocVQA answers from what it saw in training rather than from the
  layout in front of it.
- **The failure modes differ in kind, not just in rate.** The rules stayed
  silent on every field they could not resolve and were never wrong; the model
  never stayed silent and was wrong ten times out of eleven. Whether that is
  acceptable depends entirely on whether anything downstream can tolerate a
  confident wrong value.
- **Cost is not negligible**: 181 seconds per document against 7, on CPU, plus
  an 800 MB model download.

## 7. Limitations

- Only four images were compared, all from one of the nine DocXPand classes.
  The direction is consistent but the numbers are not statistically solid; week
  5 should widen the sample.
- The four images were chosen for high detection quality. That isolates the
  extraction stage, but it also means the classic pipeline is measured under
  favourable conditions, its full-dataset detection rate is 56%, so in normal
  use a further 44% of documents would never reach extraction at all.
- Only the DocVQA variant was tried in depth. A model fine-tuned on identity
  documents would be a fairer comparison, but training is outside the brief.
- Inference ran on CPU throughout. On a GPU the speed gap would narrow
  considerably, though not the accuracy gap.
- Donut returns no confidence score, so its answers cannot be filtered by
  certainty the way OCR results can, part of why its wrong answers pass
  through unflagged.