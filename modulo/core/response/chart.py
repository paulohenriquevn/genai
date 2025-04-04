import base64
import io
from typing import Any

from PIL import Image

from .base import BaseResponse


class ChartResponse(BaseResponse):
    """
    Class for handling chart/plot responses.
    """

    def __init__(self, value: Any, last_code_executed: str = None):
        """
        Initialize a chart response.
        
        :param value: The chart value (path or base64 data)
        :param last_code_executed: The code that generated this value (optional)
        """
        super().__init__(value, "chart", last_code_executed)

    def _get_image(self) -> Image.Image:
        """
        Get PIL Image from value, which can be a path or base64 string.
        
        :return: PIL Image object
        """
        if not self.value.startswith("data:image"):
            return Image.open(self.value)

        base64_data = self.value.split(",")[1]
        image_data = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_data))

    def save(self, path: str):
        """
        Save the chart image to a file.
        
        :param path: Path to save the image
        """
        img = self._get_image()
        img.save(path)

    def show(self):
        """
        Display the chart image.
        """
        img = self._get_image()
        img.show()

    def __str__(self) -> str:
        """String representation shows the image and returns the value."""
        self.show()
        return self.value

    def get_base64_image(self) -> str:
        """
        Get a base64 encoded representation of the image.
        
        :return: Base64 encoded string
        """
        img = self._get_image()
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")