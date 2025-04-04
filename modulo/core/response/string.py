from typing import Any

from .base import BaseResponse


class StringResponse(BaseResponse):
    """
    Class for handling string responses.
    """

    def __init__(self, value: Any = None, last_code_executed: str = None):
        """
        Initialize a string response.
        
        :param value: The string value
        :param last_code_executed: The code that generated this value
        """
        super().__init__(value, "string", last_code_executed)