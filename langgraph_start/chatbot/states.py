from typing import Annotated, List

from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: message history
        query: question string
        # web_search: whether to use search
        docs: list of documents
        answer: answer string
        in_stock: is product is in our stock
    """
    # web_search: str
    messages: Annotated[list, add_messages]
    query: str
    answer: str
    docs: List[str]
    iteration: int