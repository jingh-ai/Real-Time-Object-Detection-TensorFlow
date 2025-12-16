import numpy as np
from src.utils.box import xyxy_to_xywh, xywh_to_xyxy, ltwh_to_xyxy, xyxy_to_ltwh, xywh_to_ltwh, ltwh_to_xywh

class Bboxes:
    """A class for handling bounding boxes in multiple formats.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh' and provides methods for format
    conversion, scaling, and area calculation. Bounding box data should be provided as numpy arrays.
    Args:
        bboxes (np.ndarray): The bounding boxes stored in a 2D numpy array with shape (N, 4).
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Methods:
        convert: Convert bounding box format from one type to another.
        areas: Calculate the area of bounding boxes.
        mul: Multiply bounding box coordinates by scale factor(s).
        add: Add offset to bounding box coordinates.
        concatenate: Concatenate multiple Bboxes objects.

    Examples:
        Create bounding boxes in YOLO format
        >>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
        >>> bboxes.convert("xyxy")
        >>> print(bboxes.areas())
    """
    _FORMATS = ["xyxy", "xywh", "ltwh"]

    def __init__(self, bboxes: np.ndarray, format: str = "xyxy"):
        """Initialize Bbox object."""
        assert format in self._FORMATS, f"Unsupported format: {format}, formats supported: {self._FORMATS}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, target_format: str) -> 'Bbox':
        """Convert bounding box format from one type to another.

        Args:
            target_format (str): The target format to convert to ('xyxy', 'xywh', or 'ltwh').
        """
        assert target_format in self._FORMATS, f"Unsupported format: {target_format}, formats supported: {self._FORMATS}"
        if self.format == target_format:
            return
        elif self.format == "xyxy" and target_format == "xywh":
            self.bboxes = xyxy_to_xywh(self.bboxes)
        elif self.format == "xywh" and target_format == "xyxy":
            self.bboxes = xywh_to_xyxy(self.bboxes)
        elif self.format == "ltwh" and target_format == "xyxy":
            self.bboxes = ltwh_to_xyxy(self.bboxes)
        elif self.format == "xyxy" and target_format == "ltwh":
            self.bboxes = xyxy_to_ltwh(self.bboxes)
        elif self.format == "xywh" and target_format == "ltwh":
            self.bboxes = xywh_to_ltwh(self.bboxes)
        elif self.format == "ltwh" and target_format == "xywh":
            self.bboxes = ltwh_to_xywh(self.bboxes)
        self.format = target_format
    
    def areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes.

        Returns:
            np.ndarray: An array of areas for each bounding box.
        """
        if self.format == "xyxy":
            widths = self.bboxes[:, 2] - self.bboxes[:, 0]
            heights = self.bboxes[:, 3] - self.bboxes[:, 1]
        elif self.format == "xywh":
            widths = self.bboxes[:, 2]
            heights = self.bboxes[:, 3]
        elif self.format == "ltwh":
            widths = self.bboxes[:, 2]
            heights = self.bboxes[:, 3]
        return widths * heights
    
    def mul(self, scales: np.ndarray) -> None:
        """Multiply bounding box coordinates by scale factor(s).

        Args:
            scales (np.ndarray): A 1D array of scale factors for each bounding box.
        """
        assert scales.ndim == 1
        assert scales.shape[0] == self.bboxes.shape[0]
        self.bboxes *= scales[:, None]

    def add(self, offsets: np.ndarray) -> None:
        """Add offset to bounding box coordinates.

        Args:
            offsets (np.ndarray): A 2D array of offsets with shape (N, 2) for each bounding box.
        """
        assert offsets.ndim == 2
        assert offsets.shape[0] == self.bboxes.shape[0]
        assert offsets.shape[1] == 2
        if self.format == "xyxy":
            self.bboxes[:, [0, 2]] += offsets[:, 0:1]  # x offsets
            self.bboxes[:, [1, 3]] += offsets[:, 1:2]  # y offsets
        elif self.format == "xywh":
            self.bboxes[:, 0] += offsets[:, 0]  # x center offsets
            self.bboxes[:, 1] += offsets[:, 1]  # y center offsets
        elif self.format == "ltwh":
            self.bboxes[:, 0] += offsets[:, 0]  # left offsets
            self.bboxes[:, 1] += offsets[:, 1]  # top offsets   

            
