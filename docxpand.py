import json
import os

# DocXPand field name -> the name used in this project's output JSON
FIELD_MAP = {
    "family_name": "last_name",
    "given_name": "first_name",
    "birth_date": "date_of_birth",
    "birth_place": "place_of_birth",
    "date_issued": "date_of_issue",
    "expires": "date_of_expiry",
    "authority": "issued_by",
    "document_number": "document_number",
}

# fields that are images rather than text, so there is nothing to compare
NON_TEXT_FIELDS = {"photo", "ghost", "signature"}


def _first_value(value):
    """DocXPand stores some values as a single string and some as a list
    (multi-line addresses, names with several parts). Normalise to a string."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).strip() or None
    if value is None:
        return None
    return str(value).strip() or None


class GroundTruth:
    """Ground truth for the DocXPand images, indexed by filename."""

    def __init__(self, labels_path):
        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)
        self.by_filename = {}
        for document in data["documents"]:
            self.by_filename[document["filename"]] = document

    def __len__(self):
        return len(self.by_filename)

    def lookup(self, image_path):
        """Find the entry for an image path. DocXPand filenames look like
        'PP_TD3_A/<uuid>-PP_TD3_A-front.jpg', so match on the last two path
        components regardless of where the images were extracted."""
        parts = image_path.replace("\\", "/").split("/")
        key = "/".join(parts[-2:])
        return self.by_filename.get(key)

    def quad(self, image_path, image_shape):
        """The document corners in pixels, ordered p1..p4 as stored.

        DocXPand stores them normalised (0..1), so they are multiplied by the
        image size here.
        """
        entry = self.lookup(image_path)
        if entry is None:
            return None
        position = entry["annotations"][0].get("position")
        if not position:
            return None
        height, width = image_shape[:2]
        try:
            return [(float(position[p]["x"]) * width,
                     float(position[p]["y"]) * height)
                    for p in ("p1", "p2", "p3", "p4")]
        except (KeyError, TypeError, ValueError):
            return None

    def fields(self, image_path):
        """The printed field values, renamed to this project's field names."""
        entry = self.lookup(image_path)
        if entry is None:
            return {}
        raw = entry["annotations"][0].get("fields") or {}
        # fields are nested per side ("front" / "back")
        merged = {}
        for side_fields in raw.values():
            if isinstance(side_fields, dict):
                merged.update(side_fields)

        result = {}
        for docxpand_name, our_name in FIELD_MAP.items():
            field = merged.get(docxpand_name)
            if isinstance(field, dict):
                result[our_name] = _first_value(field.get("value"))
            else:
                result[our_name] = None
        return result

    def mrz(self, image_path):
        """The MRZ lines, if this document has one."""
        entry = self.lookup(image_path)
        if entry is None:
            return None
        raw = entry["annotations"][0].get("fields") or {}
        for side_fields in raw.values():
            if not isinstance(side_fields, dict):
                continue
            field = side_fields.get("mrz")
            if isinstance(field, dict) and isinstance(field.get("value"), list):
                return field["value"]
        return None

    def document_class(self, image_path):
        """The document class, e.g. 'PP_TD3_A'."""
        return image_path.replace("\\", "/").split("/")[-2]


def find_labels(search_paths=None):
    """Locate DocXPand-25k.json in the usual places."""
    candidates = search_paths or [
        "data/docxpand/DocXPand-25k/labels/DocXPand-25k.json",
        os.path.expanduser("~/docxpand/DocXPand-25k/labels/DocXPand-25k.json"),
        "DocXPand-25k.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None