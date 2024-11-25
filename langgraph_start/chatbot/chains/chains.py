from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from chatbot.llm import llm
from dotenv import load_dotenv

from chatbot.prompts import (
    VECTORSTORE_QUERY_REWRITE_PROMPT,
    RAG_PROMPT,
    RE_ASK_PROMPT,
    GENERAL_QNA_PROMPT
)

load_dotenv()

# # Post-processing
# def format_docs(docs):
#     print('##########FORMAT DOCS EXECUTED##########')
#     return "\n\n".join(doc.page_content for doc in docs)


# Vectorstore Search Question Rewriter
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VECTORSTORE_QUERY_REWRITE_PROMPT),
        (
            "human",
            "Here is the initial question: \n\n {query} \n Formulate an improved question.",
        ),
    ]
)

rag_question_rewrite_chain = re_write_prompt | llm | StrOutputParser()


# RAG CHAIN
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            RAG_PROMPT
        ),
    ]
)
rag_chain = rag_prompt | llm | StrOutputParser()


# RE-ASK Chain
re_ask_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', RE_ASK_PROMPT),
        ('human', '{query}'),
    ]
)

re_ask_chain = re_ask_prompt | llm | StrOutputParser()

# General Q&A Chain
general_qna_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERAL_QNA_PROMPT),
        ("human", "{query}")  # Placeholder for the user's query
    ]
)

general_qna_chain = general_qna_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    # print(
    #     rag_question_rewrite_chain.invoke({"question": "what is short cycling?"})
    # )
    # print(
    #     websearch_question_rewrite_chain.invoke({"question": "what is short cycling?"})
    # )
    # rag_chain.invoke({"context": docs, "question": question})
    print(
        re_ask_chain.invoke({"question": "what is short cycling?"})
    )