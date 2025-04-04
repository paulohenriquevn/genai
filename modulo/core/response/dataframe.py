from typing import Any

import pandas as pd

from .base import BaseResponse


class DataFrameResponse(BaseResponse):
    """
    Class for handling DataFrame responses.
    """

    def __init__(self, value: Any = None, last_code_executed: str = None):
        """
        Initialize a dataframe response.
        
        :param value: The DataFrame value or dict to convert to DataFrame
        :param last_code_executed: The code that generated this value
        """
        value = self.format_value(value)
        super().__init__(value, "dataframe", last_code_executed)

    def format_value(self, value):
        """
        Format the value to ensure it's a DataFrame.
        
        :param value: Value to format
        :return: Formatted value as DataFrame
        """
        return pd.DataFrame(value) if isinstance(value, dict) else value