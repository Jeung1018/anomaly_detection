from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from chatbot.llm import llm


load_dotenv()


web_search_tool = TavilySearchResults(
    max_results=2,
    # include_domains=[]
)

if __name__ == "__main__":
    print(web_search_tool.invoke({'query': "Recommend some supplements for back-muscle pain"}))