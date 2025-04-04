import re
import numpy as np
import pandas as pd

from .base import BaseResponse
from .string import StringResponse
from .number import NumberResponse
from .dataframe import DataFrameResponse
from .chart import ChartResponse
from .error import ErrorResponse


class InvalidOutputValueMismatch(Exception):
    """Exception raised when the output value doesn't match the expected type."""
    pass


class ResponseParser:
    """
    Class to parse and validate response values.
    """
    
    def parse(self, result: dict, last_code_executed: str = None) -> BaseResponse:
        """
        Parse the result dictionary and return the appropriate response object.
        
        :param result: Dictionary with 'type' and 'value' keys
        :param last_code_executed: The code that generated this result
        :return: Response object of appropriate type
        :raises InvalidOutputValueMismatch: If result format is invalid
        """
        self._validate_response(result)
        return self._generate_response(result, last_code_executed)

    def _generate_response(self, result: dict, last_code_executed: str = None):
        """
        Generate the appropriate response object based on result type.
        
        :param result: Dictionary with 'type' and 'value' keys
        :param last_code_executed: The code that generated this result
        :return: Response object of appropriate type
        :raises InvalidOutputValueMismatch: If type is invalid
        """
        if result["type"] == "number":
            return NumberResponse(result["value"], last_code_executed)
        elif result["type"] == "string":
            return StringResponse(result["value"], last_code_executed)
        elif result["type"] == "dataframe":
            return DataFrameResponse(result["value"], last_code_executed)
        elif result["type"] == "plot":
            return ChartResponse(result["value"], last_code_executed)
        else:
            raise InvalidOutputValueMismatch(f"Invalid output type: {result['type']}")

    def _validate_response(self, result: dict):
        """
        Validate that the response has the correct format and type.
        
        :param result: Dictionary with 'type' and 'value' keys
        :return: True if valid
        :raises InvalidOutputValueMismatch: If invalid
        """
        if (
            not isinstance(result, dict)
            or "type" not in result
            or "value" not in result
        ):
            raise InvalidOutputValueMismatch(
                'Result must be in the format of dictionary of type and value like `result = {"type": ..., "value": ... }`'
            )
        elif result["type"] == "number":
            if not isinstance(result["value"], (int, float, np.int64)):
                raise InvalidOutputValueMismatch(
                    "Invalid output: Expected a numeric value for result type 'number', but received a non-numeric value."
                )
        elif result["type"] == "string":
            if not isinstance(result["value"], str):
                raise InvalidOutputValueMismatch(
                    "Invalid output: Expected a string value for result type 'string', but received a non-string value."
                )
        elif result["type"] == "dataframe":
            if not isinstance(result["value"], (pd.DataFrame, pd.Series, dict)):
                raise InvalidOutputValueMismatch(
                    "Invalid output: Expected a Pandas DataFrame or Series, but received an incompatible type."
                )
        elif result["type"] == "plot":
            if not isinstance(result["value"], (str, dict)):
                raise InvalidOutputValueMismatch(
                    "Invalid output: Expected a plot save path str but received an incompatible type."
                )

            if isinstance(result["value"], dict) or (
                isinstance(result["value"], str)
                and "data:image/png;base64" in result["value"]
            ):
                return True

            path_to_plot_pattern = r"^(\/[\w.-]+)+(/[\w.-]+)*$|^[^\s/]+(/[\w.-]+)*$"
            if not bool(re.match(path_to_plot_pattern, result["value"])):
                raise InvalidOutputValueMismatch(
                    "Invalid output: Expected a plot save path str but received an incompatible type."
                )

        return True