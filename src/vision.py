"""Explainable geometry and thermal image inspection primitives using OpenCV."""
from __future__ import annotations
import cv2
import numpy as np


def mask_faces(image: np.ndarray) -> dict:
    """Blur detected faces before storage/transfer; verify misses for the installed camera."""
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("Expected a non-empty color image")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV frontal-face cascade is unavailable")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    masked = image.copy()
    boxes = []
    for x, y, w, h in faces:
        roi = masked[y:y+h, x:x+w]
        masked[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (0,0), sigmaX=15)
        boxes.append([int(x),int(y),int(w),int(h)])
    return {"image": masked, "face_boxes": boxes}


def inspect_geometry(image: np.ndarray, canny_low: int = 60, canny_high: int = 160,
                     min_area: float = 500.0) -> dict:
    """Return contours, circularity, bounding dimensions, and a configurable flag."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(smooth, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        circularity = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        parts.append({"area_px": area, "circularity": circularity, "width_px": int(w), "height_px": int(h), "bbox_xywh": [int(x),int(y),int(w),int(h)]})
    return {"edge_pixels": int(np.count_nonzero(edges)), "parts": parts, "edges": edges}


def inspect_thermal(image: np.ndarray, threshold: int = 220, min_area: int = 20) -> dict:
    """Segment bright thermal hotspots in HSV value channel; calibrate per camera/material."""
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("Expected a non-empty three-channel thermal visualization")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Default hue range targets common red/orange/yellow ironbow palettes. Select values
    # from a calibrated camera palette in production; V alone is not physical temperature.
    mask = cv2.inRange(hsv, np.array([0, 100, threshold], np.uint8), np.array([45, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspots = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area >= min_area:
            x,y,w,h = cv2.boundingRect(contour)
            hotspots.append({"area_px": area, "bbox_xywh": [int(x),int(y),int(w),int(h)]})
    return {"hotspot_count": len(hotspots), "hotspots": hotspots, "mask": mask}
