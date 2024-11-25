from chatbot.chains.routers import query_router
from chatbot.chains.graders import hallucination_grader, answer_grader


### Edges ###
def route_query(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print('---ROUTE QUERY---')
    query = state['query']
    source = query_router.invoke({'query': query})
    if source.datasource == 'web_search':
        print('---ROUTE QUERY TO WEB SEARCH---')
        return 'web_search'
    elif source.datasource == 'retrieve':
        print('---ROUTE QUERY TO VECTORSTORE---')
        return 'retrieve'
    elif source.datasource == 'api_call':
        print('---ROUTE QUERY TO API CALL---')
        return 'api_call'
    elif source.datasource == 'general_qna':
        print('---ROUTE QUERY TO GENERAL Q&A---')
        return 'general_qna'
    else:
        print('---ROUTE QUERY TO RE-ASK---')
        return 're_ask'


def decide_to_answer(state):
    """
    Determines whether to generate an answer, or re-generate a query.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print('---ASSESS GRADED DOCUMENTS---')
    in_stock = state['in_stock']
    filtered_docs = state['docs']

    if not in_stock:
        return 'guide_no_product'

    if not filtered_docs:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            '---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUERY, TRANSFORM QUERY---'
        )
        return 'transform_rag_query'
    else:
        # We have relevant documents, so generate answer
        print('---DECISION: ANSWER---')
        return 'generate_answer'


def grade_answer_v_docs_and_query(state):
    """
    Determines whether the answer is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print('---CHECK HALLUCINATIONS---')
    query = state['query']
    docs = state['docs']
    answer = state['answer']

    score = hallucination_grader.invoke(
        {'docs': docs, 'answer': answer}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == 'yes':
        print('---DECISION: ANSWER IS GROUNDED IN DOCUMENTS---')
        # Check question-answering
        print('---GRADE ANSWER vs QUERY---')
        score = answer_grader.invoke({'query': query, 'answer': answer})
        grade = score.binary_score
        if grade == 'yes':
            print('---DECISION: ANSWER ADDRESSES QUERY---')
            return 'finish'
        else:
            print('---DECISION: ANSWER DOES NOT ADDRESS QUERY---')
            return 'transform_rag_query'
    else:
        print('---DECISION: ANSWER IS NOT GROUNDED IN DOCUMENTS, RE-TRY---')
        return 'generate_answer'