from .base import BasePrompt


class CorrectOutputTypeErrorPrompt(BasePrompt):
    """Prompt to correct code when the output type is not as expected."""

    template_path = "correct_output_type_error_prompt.tmpl"

    def to_json(self):
        """
        Returns a JSON representation of the prompt with error context.
        """
        context = self.props["context"]
        code = self.props["code"]
        error = self.props["error"]
        output_type = self.props["output_type"]
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
                "exception_type": "InvalidLLMOutputType",
            },
            "config": {
                "output_type": output_type,
            },
        }