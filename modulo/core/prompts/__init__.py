from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BasePrompt
from .generate_python_code_with_sql import GeneratePythonCodeWithSQLPrompt
from .correct_execute_sql_query_usage_error_prompt import CorrectExecuteSQLQueryUsageErrorPrompt
from .correct_output_type_error_prompt import CorrectOutputTypeErrorPrompt

if TYPE_CHECKING:
    from core.agent.state import AgentState


def get_chat_prompt_for_sql(context: AgentState) -> BasePrompt:
    """Returns the appropriate prompt for generating Python code with SQL."""
    return GeneratePythonCodeWithSQLPrompt(
        context=context,
        last_code_generated=context.get("last_code_generated"),
        output_type=context.output_type,
    )


def get_correct_error_prompt_for_sql(
    context: AgentState, code: str, traceback_error: str
) -> BasePrompt:
    """Returns the appropriate prompt for correcting SQL query usage errors."""
    return CorrectExecuteSQLQueryUsageErrorPrompt(
        context=context, code=code, error=traceback_error
    )


def get_correct_output_type_error_prompt(
    context: AgentState, code: str, traceback_error: str
) -> BasePrompt:
    """Returns the appropriate prompt for correcting output type errors."""
    return CorrectOutputTypeErrorPrompt(
        context=context,
        code=code,
        error=traceback_error,
        output_type=context.output_type,
    )


__all__ = [
    "BasePrompt",
    "GeneratePythonCodeWithSQLPrompt",
    "CorrectExecuteSQLQueryUsageErrorPrompt",
    "CorrectOutputTypeErrorPrompt",
]