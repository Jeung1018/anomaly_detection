import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import END, StateGraph, START
from chatbot.states import State
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

from chatbot.nodes.nodes import (
    retrieve,
    generate_answer,
    re_ask,
    web_search,
    general_qna,
    api_call
)

from chatbot.edges.edges import (
    route_query
)


workflow = StateGraph(State)

# Define the nodes
workflow.add_node('web_search', web_search)  # web search
workflow.add_node('retrieve', retrieve)  # retrieve
workflow.add_node('generate_answer', generate_answer)  # answer
workflow.add_node('re_ask', re_ask)
workflow.add_node('api_call', api_call)
workflow.add_node('general_qna', general_qna)


# Build graph
workflow.add_conditional_edges(
    START,
    route_query,
    {
        'web_search': 'web_search',
        'retrieve': 'retrieve',
        're_ask': 're_ask',
        'api_call': 'api_call',
        'general_qna': 'general_qna'
    },
)
workflow.add_edge('web_search', 'generate_answer')
workflow.add_edge('retrieve', 'generate_answer')
workflow.add_edge('re_ask', END)
workflow.add_edge('generate_answer', END)
workflow.add_edge('general_qna', END)

# Compile
llm_app = workflow.compile()

if __name__ == '__main__':
    from pprint import pprint
    QUESTION ='How is the stock price today?'
    inputs = {'query': QUESTION}
    pprint(
        llm_app.invoke({
        'query': QUESTION
        }, {'recursion_limit': 30})
    )
    # examples:
    # 'Can I retrieve real time data?'
    # 'Is there any potential short-cycling issue in our building?'
    # 'Can you see any short cycling trends on breaker 28722?'
    # 'What are the frequent reasons for the short cycling?'

    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         # Node
    #         pprint(f'Node '{key}':')
    #         # Optional: print full state at each node
    #         # pprint.pprint(value['keys'], indent=2, width=80, depth=None)
    #     pprint('\n---\n')
    #
    # # Final generation
    # pprint(value['answer'])