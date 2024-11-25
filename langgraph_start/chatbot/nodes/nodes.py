from langchain.schema import Document

from chatbot.chains.graders import retrieval_grader
from chatbot.retrievers import retriever
from chatbot.tools import web_search_tool
from chatbot.chains.chains import (rag_chain,
                    re_ask_chain,
                    rag_question_rewrite_chain,
                    general_qna_chain
                    )
from chatbot.llm import llm

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print('---RETRIEVE---')
    query = state['query']
    iteration = state.get('iteration', 0)
    # Retrieval
    docs = retriever.invoke(query)
    return {'docs': docs, 'query': query, 'iteration': iteration}


def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print('---ANSWER---')
    query = state['query']
    docs = state.get('docs', [])
    iteration = state['iteration']
    # RAG generation
    # generation = rag_question_rewrite_chain.invoke({'context': documents, 'query': query})
    final_answer = rag_chain.invoke({'query': query, 'docs': docs})
    print(f"Generated answer: {final_answer}")
    return {'docs': docs, 'query': query, 'answer': final_answer, 'iteration': iteration}


def transform_rag_query(state):
    """
    Transform the query to produce a better query for websearch.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates query key with a re-phrased query
    """

    print('---RAG TRANSFORM QUERY---')
    query = state['query']
    docs = state['docs']
    iteration = state['iteration']
    # Re-write query
    better_query = rag_question_rewrite_chain.invoke({'query': query})
    return {'docs': docs, 'query': better_query, 'iteration': iteration}


def web_search(state):
    """
    Web search based on the re-phrased query.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print('---WEB SEARCH---')
    query = state['query']
    iteration = state.get('iteration', 0)
    # Web search
    docs = web_search_tool.invoke({'query': query})

    web_results = '\n'.join([d['content'] for d in docs])
    web_results = Document(page_content=web_results)
    docs.append(web_results)
    return {'docs': docs, 'query': query, 'iteration': iteration}


def re_ask(state):
    """
    Inform user that current query is not available.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print('---RE-ASK---')
    query = state['query']
    iteration = state.get('iteration', 0)

    final_answer = re_ask_chain.invoke({'query': query})

    return {'docs': [], 'query': query, 'answer': final_answer, 'iteration': iteration}

def api_call(state):
    """
    Inform user that current query is not available.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print('---API-CALL---')
    query = state['query']
    iteration = state.get('iteration', 0)

    final_answer = "API data retrieved (mimicking)"

    return {'docs': [], 'query': query, 'answer': final_answer, 'iteration': iteration}


def general_qna(state):
    """
    General Q&A response for user queries that do not require specific data retrieval.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with response for general Q&A
    """
    print('---GENERAL Q&A---')
    query = state['query']

    # general_qna_chain 실행
    try:
        final_answer = general_qna_chain.invoke({"query": query})  # 체인 실행
        print(f"Generated general answer: {final_answer}")
    except Exception as e:
        print(f"Error in General Q&A Chain: {e}")
        final_answer = "An error occurred while generating the answer."

    return {'query': query, 'answer': final_answer}
