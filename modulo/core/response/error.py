from .base import BaseResponse


class ErrorResponse(BaseResponse):
    """
    Class for handling error responses.
    """

    def __init__(
        self,
        value="Unfortunately, I was not able to get your answer. Please try again.",
        last_code_executed: str = None,
        error: str = None,
    ):
        """
        Initialize an error response.
        
        :param value: Error message
        :param last_code_executed: The code that generated the error
        :param error: Detailed error information
        """
        super().__init__(value, "error", last_code_executed, error)