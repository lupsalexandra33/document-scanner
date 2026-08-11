import argparse
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"

# one question per field of the shared JSON schema
FIELD_QUESTIONS = {
    "last_name": "What is the surname?",
    "first_name": "What is the given name?",
    "date_of_birth": "What is the date of birth?",
    "place_of_birth": "What is the place of birth?",
    "date_of_issue": "What is the date of issue?",
    "date_of_expiry": "What is the date of expiry?",
    "issued_by": "Which authority issued this document?",
    "document_number": "What is the document number?",
}

# answers the model returns when it has nothing — treated as "not found"
EMPTY_ANSWERS = {"", "none", "n/a", "unanswerable", "no", "unknown"}

_processor = None
_model = None


def load_model(model_name=MODEL_NAME):
    # load once and keep it: startup costs far more than a single inference
    global _processor, _model
    if _model is None:
        _processor = DonutProcessor.from_pretrained(model_name)
        _model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _model.eval()
    return _processor, _model


def ask(pixel_values, question, max_length=48):
    # ask the model one question about the image and return its answer
    processor, model = load_model()
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    decoder_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt").input_ids

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            decoder_input_ids=decoder_ids,
            max_length=max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    decoded = processor.batch_decode(output)[0]
    # keep only what follows the answer marker, drop the special tokens
    answer = re.sub(r".*<s_answer>", "", decoded, flags=re.DOTALL)
    answer = answer.replace("</s_answer>", "").replace("</s>", "").strip()
    return answer


def normalise_date(text):
    # donut writes dates as '25 01 1954'; the shared schema uses '25.01.1954'
    if not text:
        return text
    match = re.search(r'(\d{1,2})[\s./-](\d{1,2})[\s./-](\d{2,4})', text)
    if not match:
        return text
    day, month, year = match.groups()
    return f"{day.zfill(2)}.{month.zfill(2)}.{year}"


def clean(field_name, answer):
    # turn a raw answer into a field value, or None if there is nothing there
    if answer is None or answer.strip().lower() in EMPTY_ANSWERS:
        return None
    answer = answer.strip()
    if "date" in field_name:
        answer = normalise_date(answer)
    return answer or None


def process_document(image_path, fields=None):
    # run the model over one image and return the shared JSON structure
    image = Image.open(image_path).convert("RGB")
    processor, _ = load_model()
    pixel_values = processor(image, return_tensors="pt").pixel_values

    wanted = fields or list(FIELD_QUESTIONS)
    values = {name: None for name in FIELD_QUESTIONS}
    raw_answers = {}
    started = time.time()

    for name in wanted:
        raw = ask(pixel_values, FIELD_QUESTIONS[name])
        raw_answers[name] = raw
        values[name] = clean(name, raw)

    elapsed = time.time() - started
    missing = [name for name, value in values.items() if value is None]

    return {
        "document_type": None,          # the model is not asked to classify
        "fields": {**values, "personal_number": None},
        "validation": {
            "missing_fields": missing + ["personal_number"],
            "warnings": [],
            "ocr_confidence": None,  
        },
        "model": {
            "name": MODEL_NAME,
            "questions_asked": len(wanted),
            "seconds": round(elapsed, 1),
            "raw_answers": raw_answers,  # kept so hallucinations stay visible
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--fields", nargs="*", default=None,
                        help="only ask about these fields (each one is slow)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"loading {MODEL_NAME} (first run downloads ~800 MB)...")
    result = process_document(args.image, fields=args.fields)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = args.out or f"outputs_donut/{base}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out_path}")