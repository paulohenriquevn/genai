import json
from typing import Any


class BaseResponse:
    """
    Base class for different types of response values.
    """

    def __init__(
        self,
        value: Any = None,
        type: str = None,
        last_code_executed: str = None,
        error: str = None,
    ):
        """
        Initialize the BaseResponse object

        :param value: The value of the response
        :param type: The type of the response
        :param last_code_executed: The last code executed to generate the value
        :param error: Error message if any
        :raise ValueError: If value or type is None
        """
        if value is None:
            raise ValueError("Result should not be None")
        if type is None:
            raise ValueError("Type should not be None")

        self.value = value
        self.type = type
        self.last_code_executed = last_code_executed
        self.error = error

    def __str__(self) -> str:
        """Return the string representation of the response."""
        return str(self.value)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return f"{self.__class__.__name__}(type={self.type!r}, value={self.value!r})"

    def to_dict(self) -> dict:
        """Return a dictionary representation."""
        return self.__dict__

    def to_json(self) -> str:
        """Return a JSON representation."""
        return json.dumps(self.to_dict(), cls=CustomJsonEncoder)


class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special data types."""
    
    def default(self, obj):
        # Add custom serialization for special types here
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
        except ImportError:
            pass
            
        # Let the base class handle it or raise TypeError
        return super().default(obj)