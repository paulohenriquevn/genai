from typing import List, Dict, Any, Optional, Union
import pandas as pd


class AgentMemory:
    """
    Class to store agent memory including conversation history.
    """
    
    def __init__(self, agent_description: str = ""):
        self.agent_description = agent_description
        self._conversation = []
    
    def add_message(self, message: str):
        """Add a message to the conversation history."""
        self._conversation.append(message)
    
    def get_last_message(self) -> str:
        """Get the last message in the conversation."""
        if not self._conversation:
            return ""
        return self._conversation[-1]
    
    def get_conversation(self) -> str:
        """Get the full conversation history as a string."""
        return "\n".join(self._conversation)
    
    def get_previous_conversation(self) -> str:
        """Get the conversation history excluding the last message."""
        if len(self._conversation) <= 1:
            return ""
        return "\n".join(self._conversation[:-1])
    
    def count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self._conversation)
    
    def to_json(self) -> List[str]:
        """Get the conversation as a JSON-serializable list."""
        return self._conversation


class AgentConfig:
    """
    Configuration for the agent.
    """
    
    def __init__(self, direct_sql: bool = False):
        self.direct_sql = direct_sql


class AgentState:
    """
    Represents the current state of the agent including data, memory, and configuration.
    """
    
    def __init__(
        self,
        dfs: List[pd.DataFrame] = None,
        memory: AgentMemory = None,
        config: AgentConfig = None,
        output_type: str = None,
        vectorstore: Any = None,
    ):
        self.dfs = dfs or []
        self.memory = memory or AgentMemory()
        self.config = config or AgentConfig()
        self.output_type = output_type
        self.vectorstore = vectorstore
        self._state = {}
    
    def set(self, key: str, value: Any):
        """Set a value in the state dictionary."""
        self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state dictionary."""
        return self._state.get(key, default)
    
    def add_df(self, df: pd.DataFrame):
        """Add a dataframe to the state."""
        self.dfs.append(df)