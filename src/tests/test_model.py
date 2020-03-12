from PIL import Image

import pytest
from src.yolo import Yolo


class TestDetection():

    @pytest.fixture(autouse=True)
    def setUpClass(self):
        self.yolo = Yolo()

    def test_detect(self):
        image = Image.open('images/1.jpg')
        result = self.yolo.detect_image(image, draw_bbox=False)
        assert len(result) == 3
        assert result[1].item() > 0.9
        assert result[2].item() == 2

        result = self.yolo.detect_image(image)
