# Week 4 — Pipeline 2: a pretrained document-understanding model

Pipeline 1 detects the document, runs OCR and applies hand-written rules.
Pipeline 2 hands the image to a pretrained model and reads structured data out
of it — no detection, no regex, no layout rules. Both write the same JSON
schema, so their results can be compared directly.

Per the brief, the model is used as-is: no training and no fine-tuning. What it
produces is recorded as it comes, mistakes included.

## 1. Choosing a variant

**`donut-base`** — the plain pretrained backbone. On an identity document it
degenerates: the decoder emits the same token repeatedly.

```
<s_cord-v2> I D D D D D D D D D D D D D D D D D D D D D D D ...
```

This is not a bug in the integration. A pretrained backbone has no task; without
fine-tuning on something, it has no output format to produce. It was discarded.

**`donut-base-finetuned-docvqa`** — fine-tuned for document visual question
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

Both pipelines were run over the same DocXPand images, scored against the same
ground truth with the same comparison function (case, accents and punctuation
normalised).

| pipeline | correct | wrong | missing | accuracy | avg time |
|----------|---------|-------|---------|----------|----------|
| classic (OCR + rules) | 4 | **0** | 7 | **36%** | 7 s |
| Donut (DocVQA) | 1 | 2 | 8 | 9% | 78 s |

The classic pipeline is four times more accurate and eleven times faster.

The difference in *failure behaviour* matters more than the difference in
accuracy. The classic pipeline made no wrong statements at all — when a rule
does not match, the field stays `null`. Donut answered three times and was wrong
twice. For a document reader that distinction is important: a missing field is
visible and can be re-read, a wrong one passes unnoticed.

## 4. What the model actually gets wrong

The most informative single case, with ground truth alongside:

| field | ground truth | Donut answered |
|-------|--------------|----------------|
| document_number | XQ6D4PW94 | `xq6d4pw94` — correct |
| last_name | Maréchal-Dubois | `xq6d4pw948pil5011287` — the MRZ string |
| date_of_birth | 28.11.1950 | `marchal-dubois` — the surname |

This is not blind hallucination. Every value the model returned is really
printed on the document — it read the text correctly and attached it to the
wrong field. Asked for the surname it returned the MRZ; asked for the birth date
it returned the surname.

An earlier single-image test produced a different failure mode — asked for the
surname on another document it answered `cigarvival`, a word that appears
nowhere on it. So both behaviours occur: mostly field misalignment, occasionally
invented text.

**The model sees the document but does not understand what is being asked of
it.** That is a more specific diagnosis than "the model is wrong", and it points
somewhere: fine-tuning on identity documents would likely fix the alignment,
because the reading is already there. A model that could not read the text at
all would be a much harder problem.

## 5. Sensitivity to input

Two of the three test images were the back of an ID card, where ground truth
covers only issue date, expiry and authority. Donut returned nothing usable for
either — all fields missing. The front, which carries the photo, the MRZ and
most printed fields, is where it produced answers.

This matches what the classic pipeline shows on the detection side (fronts
detected at 71%, backs at 32%): the back of a document is the harder case for
both approaches, because it carries fewer distinctive visual features.

## 6. Conclusions

- **A pretrained backbone alone is unusable.** `donut-base` produces degenerate
  output on this domain; a task-fine-tuned variant is the minimum.
- **The modern pipeline loses to hand-written rules here** — 9% against 36% —
  which is the opposite of the expected result. Without fine-tuning on identity
  documents, DocVQA answers from what it saw in training rather than from the
  layout in front of it.
- **The failure modes differ in kind, not just in rate.** Rules stay silent when
  they do not match; the model answers anyway. Which is preferable depends on
  whether a downstream consumer can tolerate a confident wrong value.
- **Cost is not negligible**: 78 seconds against 7, on CPU, plus an 800 MB model
  download.

## 7. Limitations

- Only three images were compared, on two of nine DocXPand classes. The
  direction is clear but the numbers are not statistically solid; week 5 should
  widen the sample.
- Only the DocVQA variant was tried in depth. A model fine-tuned on identity
  documents would be a fairer comparison, but training is outside the brief.
- Inference ran on CPU throughout. On a GPU the speed gap would narrow
  considerably, though not the accuracy gap.
- Donut returns no confidence score, so its answers cannot be filtered by
  certainty the way OCR results can — part of why its wrong answers pass
  through unflagged.