from typing import Any

from .base import BaseResponse


class NumberResponse(BaseResponse):
    """
    Class for handling numerical responses.
    """

    def __init__(self, value: Any = None, last_code_executed: str = None):
        """
        Initialize a number response.
        
        :param value: The numerical value
        :param last_code_executed: The code that generated this value
        """
        super().__init__(value, "number", last_code_executed)