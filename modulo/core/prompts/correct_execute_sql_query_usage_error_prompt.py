from .base import BasePrompt


class CorrectExecuteSQLQueryUsageErrorPrompt(BasePrompt):
    """Prompt to correct code when SQL query function is not properly used."""

    template_path = "correct_execute_sql_query_usage_error_prompt.tmpl"

    def to_json(self):
        """
        Returns a JSON representation of the prompt with error context.
        """
        context = self.props["context"]
        code = self.props["code"]
        error = self.props["error"]
        memory = context.memory
        conversations = memory.to_json()

        system_prompt = memory.agent_description

        # prepare datasets for context
        datasets = [dataset.to_json() for dataset in context.dfs]

        return {
            "datasets": datasets,
            "conversation": conversations,
            "system_prompt": system_prompt,
            "error": {
                "code": code,
                "error_trace": str(error),
                "exception_type": "ExecuteSQLQueryNotUsed",
            },
        }