from .base import BasePrompt


class GenerateSystemMessagePrompt(BasePrompt):
    """Prompt to generate the system message for the conversation."""

    template_path = "generate_system_message.tmpl"