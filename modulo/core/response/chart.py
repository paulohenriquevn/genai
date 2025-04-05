import base64
import io
import json
from typing import Any, Dict, Optional, Union

from PIL import Image

from .base import BaseResponse


class ChartResponse(BaseResponse):
    """
    Class for handling chart/plot responses.
    
    Supports multiple chart formats:
    - 'image': Traditional matplotlib image format (default)
    - 'apex': ApexCharts JSON configuration for interactive web charts
    """

    def __init__(self, value: Any, chart_format: str = "image", last_code_executed: str = None):
        """
        Initialize a chart response.
        
        Args:
            value: The chart value (path, base64 data, or ApexCharts config)
            chart_format: Format of the chart ('image' or 'apex')
            last_code_executed: The code that generated this value (optional)
        """
        super().__init__(value, "chart", last_code_executed)
        self.chart_format = chart_format
        
        # Validate chart format
        if chart_format not in ["image", "apex"]:
            raise ValueError(f"Unsupported chart format: {chart_format}. Use 'image' or 'apex'.")

    def _get_image(self) -> Optional[Image.Image]:
        """
        Get PIL Image from value, which can be a path or base64 string.
        Only applicable for 'image' format.
        
        Returns:
            PIL Image object or None if format is not 'image'
        """
        if self.chart_format != "image":
            return None
            
        if not isinstance(self.value, str):
            raise ValueError("Expected string value for image format charts")
            
        if not self.value.startswith("data:image"):
            return Image.open(self.value)

        base64_data = self.value.split(",")[1]
        image_data = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_data))

    def save(self, path: str):
        """
        Save the chart to a file.
        For 'image' format, saves as an image.
        For 'apex' format, saves as a JSON file.
        
        Args:
            path: Path to save the chart
        """
        if self.chart_format == "image":
            img = self._get_image()
            if img:
                img.save(path)
        elif self.chart_format == "apex":
            with open(path, 'w', encoding='utf-8') as f:
                if isinstance(self.value, Dict):
                    json.dump(self.value, f, indent=2)
                else:
                    # Assume it's already a JSON string
                    f.write(self.value)

    def show(self):
        """
        Display the chart. 
        For 'image' format, shows the image.
        For 'apex' format, prints the JSON configuration.
        """
        if self.chart_format == "image":
            img = self._get_image()
            if img:
                img.show()
        elif self.chart_format == "apex":
            if isinstance(self.value, Dict):
                print(json.dumps(self.value, indent=2))
            else:
                # Assume it's already a JSON string
                print(self.value)

    def __str__(self) -> str:
        """String representation based on format."""
        if self.chart_format == "image":
            self.show()
            return self.value
        elif self.chart_format == "apex":
            return f"ApexCharts configuration (format: {self.chart_format})"

    def get_base64_image(self) -> Optional[str]:
        """
        Get a base64 encoded representation of the image.
        Only applicable for 'image' format.
        
        Returns:
            Base64 encoded string or None if format is not 'image'
        """
        if self.chart_format != "image":
            return None
            
        img = self._get_image()
        if not img:
            return None
            
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")
    
    def to_apex_json(self) -> Optional[Dict[str, Any]]:
        """
        Get the ApexCharts configuration as a Python dictionary.
        Only applicable for 'apex' format.
        
        Returns:
            Dictionary with ApexCharts configuration or None if format is not 'apex'
        """
        if self.chart_format != "apex":
            return None
            
        if isinstance(self.value, Dict):
            return self.value
        else:
            # Assume it's a JSON string and parse it
            try:
                return json.loads(self.value)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("Invalid ApexCharts configuration format")
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the chart to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the chart
        """
        result = {
            "type": "chart",
            "format": self.chart_format
        }
        
        if self.chart_format == "image":
            # For image format, include the path or base64 data
            result["value"] = self.value
        elif self.chart_format == "apex":
            # For apex format, include the configuration
            if isinstance(self.value, Dict):
                result["config"] = self.value
            else:
                # Try to parse JSON string
                try:
                    result["config"] = json.loads(self.value)
                except (json.JSONDecodeError, TypeError):
                    result["config"] = self.value  # Keep as is if parsing fails
        
        return result