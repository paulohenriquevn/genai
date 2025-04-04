class BaseCoreException(Exception):
    """Base exception for all custom exceptions in the core module."""
    pass


class InvalidOutputValueMismatch(BaseCoreException):
    """
    Exception raised when the output value doesn't match the expected type.
    """
    pass


class ExecuteSQLQueryNotUsed(BaseCoreException):
    """
    Exception raised when the code should use execute_sql_query but doesn't.
    """
    pass


class InvalidLLMOutputType(BaseCoreException):
    """
    Exception raised when the LLM output is not of the expected type.
    """
    pass


class UnknownLLMOutputType(BaseCoreException):
    """
    Exception raised when the LLM output type is not recognized.
    """
    pass


class TemplateRenderError(BaseCoreException):
    """
    Exception raised when a template cannot be rendered.
    """
    pass