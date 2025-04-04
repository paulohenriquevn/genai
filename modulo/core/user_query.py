class UserQuery:
    """
    Represents a user query or question in natural language.
    """
    
    def __init__(self, user_query: str):
        """
        Initialize a user query.
        
        :param user_query: The natural language query from the user
        """
        self.value = user_query

    def __str__(self):
        """String representation of the query."""
        return self.value

    def __repr__(self):
        """Detailed string representation for debugging."""
        return f"UserQuery(value={self.value!r})"

    def to_dict(self):
        """Dictionary representation of the query."""
        return {"query": self.value}

    def to_json(self):
        """JSON representation of the query."""
        return self.value