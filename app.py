"""
    streamlit run app.py
"""
import json
import time

import cv2
import numpy as np
import streamlit as st

import preprocessing

st.set_page_config(page_title="Document Scanner", layout="wide",
                   page_icon="🌸")

# Palette. The theme in .streamlit/config.toml colours Streamlit's own widgets;
# these are for the elements drawn by hand below.
PINK = "#d6336c"
PINK_SOFT = "#ffe3ec"
PINK_DEEP = "#a61e4d"
INK = "#3d2b33"
MUTED = "#9c7d88"

st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {PINK_DEEP}; letter-spacing: -0.01em; }}
  h1 {{ font-weight: 700; }}

  /* field rows: one card each, so an empty field reads as clearly empty */
  .field-card {{
      background: white;
      border: 1px solid {PINK_SOFT};
      border-left: 4px solid {PINK};
      border-radius: 10px;
      padding: 0.55rem 0.9rem;
      margin-bottom: 0.4rem;
  }}
  .field-card.empty {{ border-left-color: {MUTED}; opacity: 0.75; }}
  .field-label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: {MUTED};
      margin-bottom: 0.1rem;
  }}
  .field-value {{ font-size: 1rem; color: {INK}; font-weight: 600; }}
  .field-value.empty {{ font-weight: 400; font-style: italic; color: {MUTED}; }}

  /* rounded images, so screenshots look composed rather than pasted */
  [data-testid="stImage"] img {{
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(214, 51, 108, 0.12);
  }}

  [data-testid="stSidebar"] {{ border-right: 1px solid {PINK_SOFT}; }}
  .stDownloadButton button {{ border-radius: 999px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classic():
    # import lazily and cache: EasyOCR takes several seconds to start
    import extract
    return extract


@st.cache_resource
def load_donut():
    import donut_pipeline
    return donut_pipeline


def to_rgb(image):
    # OpenCV works in BGR; Streamlit expects RGB
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_detection(image, quad):
    # outline the detected document on a copy of the original
    preview = image.copy()
    if quad is not None:
        import scanner
        points = np.int32(scanner.order_points(quad.reshape(4, 2)))
        cv2.polylines(preview, [points], True, (0, 255, 0),
                      max(2, image.shape[1] // 300))
    return preview


def show_fields(fields):
    # render the extracted fields as cards
    for name, value in fields.items():
        label = name.replace("_", " ")
        if value:
            st.markdown(
                f'<div class="field-card">'
                f'<div class="field-label">{label}</div>'
                f'<div class="field-value">{value}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="field-card empty">'
                f'<div class="field-label">{label}</div>'
                f'<div class="field-value empty">not extracted</div></div>',
                unsafe_allow_html=True)


st.title("🌸 Document Scanner & Verifier")
st.caption("Classic CV + OCR + rules, compared against a pretrained "
           "document-understanding model")
st.divider()

with st.sidebar:
    st.header("Settings")

    pipeline_choice = st.radio(
        "Pipeline",
        ["Classic (OCR + rules)", "Pretrained model (Donut)", "Both"],
        help="The classic pipeline runs in seconds. Donut needs roughly "
             "25 seconds per field on CPU.",
    )

    variant = st.selectbox(
        "Preprocessing variant",
        list(preprocessing.PREPROCESSING_VARIANTS),
        index=list(preprocessing.PREPROCESSING_VARIANTS).index("warped"),
        help="What the OCR engine is shown. 'raw' skips geometric correction "
             "entirely — on some datasets that works better.",
    )

    donut_fields = None
    if pipeline_choice != "Classic (OCR + rules)":
        st.warning("Donut is slow on CPU and needs about 1 GB of memory. "
                   "Limit the fields to keep it usable.")
        # the field list is hard-coded rather than imported from
        # donut_pipeline, so that merely opening the sidebar does not load an
        # 800 MB model into memory
        donut_fields = st.multiselect(
            "Fields to ask Donut about",
            ["last_name", "first_name", "date_of_birth", "place_of_birth",
             "date_of_issue", "date_of_expiry", "issued_by", "document_number"],
            default=["date_of_birth", "document_number"],
        )

uploaded = st.file_uploader("Upload a document image",
                            type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an identity document to begin. The sample images under "
            "`outputs_week5/` correspond to documents from the DocXPand test "
            "set.")
    st.stop()

data = np.frombuffer(uploaded.read(), np.uint8)
image = cv2.imdecode(data, cv2.IMREAD_COLOR)
if image is None:
    st.error("Could not read that file as an image.")
    st.stop()

# stage 1: localisation
st.header("① Document localisation")
quad, method = preprocessing.detect_document_improved(image)

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(to_rgb(draw_detection(image, quad)),
             caption=f"{image.shape[1]} x {image.shape[0]}")
with right:
    st.subheader("What OCR receives")
    prepared = preprocessing.PREPROCESSING_VARIANTS[variant](image, quad)
    st.image(to_rgb(prepared),
             caption=f"variant: {variant} — {prepared.shape[1]} x {prepared.shape[0]}")

if quad is None:
    st.warning("No document located. The pipeline falls back to the whole "
               "image — which, on frontally photographed documents, is often "
               "the better OCR input anyway.")
else:
    st.success(f"Document located via **{method}** "
               f"({'clean four-corner contour' if method == 'contour' else 'rotated-rectangle fallback'})")

# stage 2: extraction
st.header("② Extraction")

run_classic = pipeline_choice in ("Classic (OCR + rules)", "Both")
run_donut = pipeline_choice in ("Pretrained model (Donut)", "Both")

if run_classic and run_donut:
    st.info("Running both loads two models at once — around 2 GB of memory. "
            "On a machine with 8 GB this can fail; run them one at a time if "
            "the app is killed.")

# write the upload somewhere the pipelines can read it
temp_path = "_uploaded_input.png"
cv2.imwrite(temp_path, image)

columns = st.columns(2 if (run_classic and run_donut) else 1)
column_index = 0

if run_classic:
    with columns[column_index]:
        st.subheader("Classic pipeline")
        with st.spinner("Running OCR and extraction rules..."):
            started = time.time()
            result = load_classic().process_document(temp_path, variant=variant)
            elapsed = time.time() - started

        st.caption(f"{elapsed:.1f} s")
        show_fields(result["fields"])

        validation = result["validation"]
        if validation["warnings"]:
            st.subheader("Validation warnings")
            for warning in validation["warnings"]:
                st.warning(warning)
        else:
            st.info("No validation warnings.")

        mrz_info = result.get("mrz", {})
        if mrz_info.get("found"):
            checks = mrz_info.get("check_digits") or {}
            passed = sum(1 for v in checks.values() if v is True)
            st.success(f"MRZ found ({mrz_info['format']}), "
                       f"{passed}/{len(checks)} check digits passed")
        else:
            st.caption("No machine-readable zone found in the OCR output.")

        with st.expander("Raw OCR text"):
            st.code("\n".join(result.get("ocr_texts", [])) or "(nothing read)")

        with st.expander("JSON"):
            st.json(result)

        st.download_button("Download JSON",
                           json.dumps(result, indent=2, ensure_ascii=False),
                           file_name="classic_output.json", mime="application/json")
    column_index += 1

if run_donut:
    with columns[column_index]:
        st.subheader("Pretrained model")
        if not donut_fields:
            st.info("Select at least one field in the sidebar.")
        else:
            estimate = 25 * len(donut_fields)
            with st.spinner(f"Asking the model {len(donut_fields)} question(s) "
                            f"— roughly {estimate} s on CPU..."):
                started = time.time()
                donut_result = load_donut().process_document(
                    temp_path, fields=donut_fields)
                elapsed = time.time() - started

            st.caption(f"{elapsed:.1f} s")
            show_fields({k: v for k, v in donut_result["fields"].items()
                         if k in donut_fields})

            st.caption("The model returns no confidence score, so its answers "
                       "cannot be filtered by certainty.")

            with st.expander("Raw answers"):
                st.json(donut_result["model"].get("raw_answers", {}))

            with st.expander("JSON"):
                st.json(donut_result)

            st.download_button("Download JSON",
                               json.dumps(donut_result, indent=2, ensure_ascii=False),
                               file_name="donut_output.json",
                               mime="application/json")

if run_classic and run_donut:
    st.header("③ Side by side")
    st.markdown(
        "Measured over a fixed 20-document test set, the classic pipeline "
        "answers 30% of the time and is right 87% of the time it answers. "
        "The model answers every time and was wrong on 18 of 20 fields — it "
        "reads the document correctly but attaches values to the wrong fields. "
        "For a document reader, knowing when it does not know matters more "
        "than raw accuracy."
    )
