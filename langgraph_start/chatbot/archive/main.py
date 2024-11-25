from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import operator
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model with your API key
llm = ChatOpenAI(api_key=api_key, model="gpt-4-turbo")

# Define the overall public state with reducers for fields that may receive multiple updates
class OverallState(TypedDict):
    query_input: str
    routing_decision: str
    # final_response can collect responses from multiple nodes if needed, using list and a reducer
    final_response: Annotated[list[str], operator.add]

# Define the Supervisor Node
def supervisor(state: OverallState) -> OverallState:
    query = state["query_input"]
    if "api" in query.lower() or "fetch" in query.lower():
        state["routing_decision"] = "route_to_api"
    else:
        state["routing_decision"] = "route_to_qa"
    print(f"Entered `supervisor`:\n\tInput: {state}.\n\tDecision: {state['routing_decision']}")
    return state

# Routing Node to General Q&A
def route_to_qa(state: OverallState) -> OverallState:
    state["query_input"] = state["query_input"]  # Pass query to the QA agent
    print(f"Routed to `general_qa_agent` with query: {state['query_input']}")
    return state

# Routing Node to API Call Agent
def route_to_api(state: OverallState) -> OverallState:
    state["query_input"] = state["query_input"]  # Pass query to the API agent
    print(f"Routed to `api_call_agent` with query: {state['query_input']}")
    return state

# Define the General Q&A Agent Node
def general_qa_agent(state: OverallState) -> OverallState:
    response = llm.invoke([HumanMessage(content=state["query_input"])])
    state["final_response"].append(response.content)  # Append response using list
    print(f"Entered `general_qa_agent`:\n\tResponse: {state['final_response']}")
    return state

# Define the API Call Agent Node
def api_call_agent(state: OverallState) -> OverallState:
    state["final_response"].append("API Calling")  # Append response using list
    print(f"Entered `api_call_agent`:\n\tResponse: {state['final_response']}")
    return state

# Construct the graph with separate routing nodes
graph = StateGraph(OverallState)
graph.add_node("supervisor", supervisor)
graph.add_node("route_to_qa", route_to_qa)
graph.add_node("route_to_api", route_to_api)
graph.add_node("general_qa_agent", general_qa_agent)
graph.add_node("api_call_agent", api_call_agent)

# Define the sequence of execution without conditions
graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", "route_to_qa")
graph.add_edge("supervisor", "route_to_api")
graph.add_edge("route_to_qa", "general_qa_agent")
graph.add_edge("route_to_api", "api_call_agent")
graph.add_edge("general_qa_agent", END)
graph.add_edge("api_call_agent", END)

# Compile the graph
compiled_graph = graph.compile()

# Function to handle user input
def main():
    user_input = input("Enter your query: ")
    initial_state = {"query_input": user_input, "final_response": []}  # Initialize final_response as a list
    final_state = compiled_graph.invoke(initial_state)
    print("\nFinal Responses:", final_state["final_response"])

if __name__ == "__main__":
    main()