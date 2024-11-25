import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.environ.get('INDEX_NAME')

# Vectorstore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=OpenAIEmbeddings(),
)

# Retriever with config (currently 2 docs)
retriever = vectorstore.as_retriever(search_kwargs={
    'k': 2,
})


if __name__ == "__main__":
    from pprint import pprint
    print('TEST: Retriever')
    docs = retriever.invoke('panel 2014 energy')
    pprint(docs)