import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # sum x+y for each point
    s = pts.sum(axis=1)
    # smallest x+y (top-left)
    rect[0] = pts[np.argmin(s)]
    # largest x+y (bottom-right)
    rect[2] = pts[np.argmax(s)]
    # difference y-x for each point
    d = np.diff(pts, axis=1)
    # smallest y-x (top-right)
    rect[1] = pts[np.argmin(d)]
    # largest y-x (bottom-left)
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(image, pts):
    # we obtain the frontal view with this function
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # output width
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    # output height
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    # the 4 destination corners
    dst = np.array([[0, 0], [maxW - 1, 0],
                    [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

    # transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def detect_document(image, resize_height=500):
    # how much we shrink by
    ratio = image.shape[0] / float(resize_height)
    # small copy
    small = cv2.resize(image, (int(image.shape[1] / ratio), resize_height))

    # grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # detect edges + dilate
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # keep only the 5 largest contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # area of the small image
    img_area = small.shape[0] * small.shape[1]
    for c in cnts:
        # perimeter length of the contour
        peri = cv2.arcLength(c, True)
        # simplify the contour to fewer points
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(c) > 0.15 * img_area:
            # reshape
            pts = approx.reshape(4, 2).astype("float32")

            return pts * ratio
    return None


# enhancement
def enhance(warped):
    # grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # enhancer
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray = clahe.apply(gray)

    # remove noise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    binar = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10)
    return binar


# complete pipeline
def scan(image):
    # find the document's 4 corners
    quad = detect_document(image)

    if quad is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray, None, False

    # straighten the document
    warped = four_point_transform(image, quad)
    # clean it up
    result = enhance(warped)
    return result, quad, True
