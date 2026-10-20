import numpy as np
import pytest
from src.vision import inspect_geometry, inspect_thermal, mask_faces

def test_geometry_detects_synthetic_component():
    import cv2
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 60, (255,255,255), 5)
    result = inspect_geometry(image, min_area=100)
    assert result["edge_pixels"] > 0
    assert len(result["parts"]) >= 1

def test_thermal_hotspot_and_invalid_input():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:40, 20:40] = (0,0,255)
    assert inspect_thermal(image, threshold=200)["hotspot_count"] == 1
    with pytest.raises(ValueError): inspect_thermal(np.zeros((0,0,3), dtype=np.uint8))

def test_face_mask_pipeline_on_empty_scene():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    result = mask_faces(image)
    assert result["image"].shape == image.shape
    assert result["face_boxes"] == []
