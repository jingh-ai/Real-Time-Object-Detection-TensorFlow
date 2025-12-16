import numpy as np


def xyxy_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x1, y1, x2, y2) to (x, y, width, height) format."""
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    bboxes_xywh = np.empty_like(bboxes)  # faster than clone/copy
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes_xywh[..., 0] = (x1 + x2) / 2  # x center
    bboxes_xywh[..., 1] = (y1 + y2) / 2  # y center
    bboxes_xywh[..., 2] = x2 - x1  # width
    bboxes_xywh[..., 3] = y2 - y1  # height
    return bboxes_xywh

def xywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x, y, width, height) to (x1, y1, x2, y2) format."""
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    bboxes_xyxy = np.empty_like(bboxes)  # faster than clone/copy
    x, y, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes_xyxy[..., 0] = x - w / 2  # x1
    bboxes_xyxy[..., 1] = y - h / 2  # y1
    bboxes_xyxy[..., 2] = x + w / 2  # x2
    bboxes_xyxy[..., 3] = y + h / 2  # y2
    return bboxes_xyxy

def ltwh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (left, top, width, height) to (x1, y1, x2, y2) format."""
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    bboxes_xyxy = np.empty_like(bboxes)  # faster than clone/copy
    l, t, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes_xyxy[..., 0] = l  # x1
    bboxes_xyxy[..., 1] = t  # y1
    bboxes_xyxy[..., 2] = l + w  # x2
    bboxes_xyxy[..., 3] = t + h  # y2
    return bboxes_xyxy

def xyxy_to_ltwh(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x1, y1, x2, y2) to (left, top, width, height) format."""
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    bboxes_ltwh = np.empty_like(bboxes)  # faster than clone/copy
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes_ltwh[..., 0] = x1  # left
    bboxes_ltwh[..., 1] = y1  # top
    bboxes_ltwh[..., 2] = x2 - x1  # width
    bboxes_ltwh[..., 3] = y2 - y1  # height
    return bboxes_ltwh

def xywh_to_ltwh(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x, y, width, height) to (left, top, width, height) format."""
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    bboxes_ltwh = np.empty_like(bboxes)  # faster than clone/copy
    x, y, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes_ltwh[..., 0] = x - w / 2  # left
    bboxes_ltwh[..., 1] = y - h / 2  # top
    bboxes_ltwh[..., 2] = w  # width
    bboxes_ltwh[..., 3] = h  # height
    return bboxes_ltwh

def ltwh_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (left, top, width, height) to (x, y, width, height) format."""
    assert bboxes.ndim == 2 and bboxes.shape[1] == 4
    bboxes_xywh = np.empty_like(bboxes)  # faster than clone/copy
    l, t, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes_xywh[..., 0] = l + w / 2  # x center
    bboxes_xywh[..., 1] = t + h / 2  # y center
    bboxes_xywh[..., 2] = w  # width
    bboxes_xywh[..., 3] = h  # height
    return bboxes_xywh



