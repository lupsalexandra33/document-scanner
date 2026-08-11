import cv2
import numpy as np
import scanner

# ID-1 card format (85.6 x 54 mm) -> expected aspect ratio of a straightened card
ID_ASPECT_RATIO = 85.6 / 54.0     # 1.585


def quad_is_plausible(quad, image_shape, min_frac=0.06, max_frac=0.90,
                      aspect_tolerance=0.45):
    """Reject detections that cannot be a document."""

    q = quad.reshape(4, 2).astype("float32")
    frac = cv2.contourArea(q) / (image_shape[0] * image_shape[1])
    if not (min_frac < frac < max_frac):
        return False

    rect = scanner.order_points(q)
    tl, tr, br, bl = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if height == 0:
        return False

    aspect = width / height
    if aspect < 1:
        aspect = 1 / aspect       # treat portrait and landscape the same
    return abs(aspect - ID_ASPECT_RATIO) <= aspect_tolerance


def detect_document_improved(image, resize_height=500, area_frac=0.08,
                             morph_close=True, use_fallback=True):
    # improved document detection

    ratio = image.shape[0] / float(resize_height)
    small = cv2.resize(image, (int(image.shape[1] / ratio), resize_height))

    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edged = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), 1)
    if morph_close:
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
    img_area = small.shape[0] * small.shape[1]

    # pass 1: a clean quadrilateral contour
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) > area_frac * img_area:
            quad = approx.reshape(4, 2).astype("float32")
            if quad_is_plausible(quad, small.shape):
                return quad * ratio, "contour"

    # pass 2: rotated bounding box of the largest plausible contour
    if use_fallback:
        for c in cnts:
            if cv2.contourArea(c) > area_frac * img_area:
                box = cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")
                if quad_is_plausible(box, small.shape):
                    return box * ratio, "minarea"

    return None, None

# OCR input variants, used by evaluate.py to compare which one OCR reads best

def variant_raw(image, quad):
    # no geometric correction at all, the original frame
    return image


def variant_warped(image, quad):
    # perspective-corrected document, still in colour
    if quad is None:
        return image
    return scanner.four_point_transform(image, quad)


def variant_binarized(image, quad):
    # perspective-corrected + the full enhance() chain
    return scanner.enhance(variant_warped(image, quad))


def variant_upscaled(image, quad, factor=3):
    # perspective-corrected then upscaled, larger glyphs for the OCR engine
    warped = variant_warped(image, quad)
    return cv2.resize(warped, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def variant_clahe_only(image, quad):
    # perspective-corrected + local contrast, but no binarization
    warped = variant_warped(image, quad)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


PREPROCESSING_VARIANTS = {
    "raw": variant_raw,
    "warped": variant_warped,
    "binarized": variant_binarized,
    "upscaled_3x": variant_upscaled,
    "clahe_only": variant_clahe_only,
}