from dotenv import load_dotenv
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from chatbot.llm import llm
from chatbot.prompts import ROUTER_SYSTEM_PROMPT

load_dotenv()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["retrieve", "general_qna", "api_call", "re_ask"] = Field(
        ...,
        description="Given a user question, choose to route it to vectorstore, general_qna, api_call, or re_ask.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

route_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', ROUTER_SYSTEM_PROMPT),
        ('human', '{query}'),
    ]
)

query_router = route_prompt | structured_llm_router


if __name__ == '__main__':
    # Test queries for each route
    print(query_router.invoke({"query": "Can you retrieve the cycle count from the power data?"}))  # Should route to vectorstore
    print(query_router.invoke({"query": "How does short-cycling impact power systems?"}))  # Should route to general_qna
    print(query_router.invoke({"query": "Please fetch the latest power data."}))  # Should route to api_call
    print(query_router.invoke({"query": "Can you recommend something for feeling tired?"}))  # Should route to re_ask
