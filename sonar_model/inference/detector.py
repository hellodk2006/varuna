from pathlib import Path

from ultralytics import YOLO


class SonarDetector:
    """
    YOLO-based detector for underwater side-scan sonar imagery.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: int | str = 0,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def predict(self, image):
        """
        Run YOLO inference on an image.

        Parameters
        ----------
        image:
            Image path or NumPy image.

        Returns
        -------
        list
            Ultralytics Results objects.
        """

        return self.model.predict(
            source=image,
            imgsz=640,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )