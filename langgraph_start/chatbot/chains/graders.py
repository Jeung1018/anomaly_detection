from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from chatbot.prompts import (
    DOCUMENT_RELEVANCE_GRADE_SYSTEM_PROMPT,
    HALLUCINATION_GRADE_SYSTEM_PROMPT,
    ANSWER_GRADE_SYSTEM_PROMPT
)
from chatbot.llm import llm
from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_GRADE_SYSTEM_PROMPT),
        ("human", "User question: \n\n {query} \n\n LLM answer: {answer}"),
    ]
)
answer_grader = answer_prompt | structured_llm_grader


# Grade Documents Relevance
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DOCUMENT_RELEVANCE_GRADE_SYSTEM_PROMPT),
        ("human", "Retrieved document: \n\n {doc} \n\n User question: {query}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader


# Grade Hallucination With Documents
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_GRADE_SYSTEM_PROMPT),
        ("human", "Set of facts: \n\n {docs} \n\n LLM answer: {answer}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader


if __name__ == "__main__":
    from llm_graph.src.chatbot.retrievers import retriever

    question = "men's fatigue related supplements"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))