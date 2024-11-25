import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index
from openai import OpenAI

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI and Pinecone
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect or create Pinecone index
index_name = "anomaly"
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1536,  # Make sure this matches your embedding dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="us-west1")
    )
index = pinecone_client.Index(index_name)

# Function to retrieve embeddings
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    # Access the embedding directly from the response
    return response.data[0].embedding

# 샘플 데이터
texts = [
    """
     --- Comprehensive Report ---
    
    Report:
    Based on the summarized data provided, a comprehensive analysis was conducted on the cycle data summaries of various breakers in different panels from 202
    4-10-02T00:00:00Z to 2024-10-03T00:00:00Z.
    
    Observations:
    
    1. The breaker 28722 in panel 1937 recorded 39 cycles. This is a moderate number of cycles and may not be indicative of any substantial issue. However, co
    ntinuous monitoring is recommended. [Visualization URL](https://app.verdigris.co/data/org/8/building/530/panel/1937/breaker/28722?timespanstart=2024-10-02
    T00%3A00%3A00Z&timespanend=2024-10-03T00%3A00%3A00Z&timeZone=UTC&resolution=1m&view=sum&timezonemode=UTC)
    
    2. Breakers 28726 and 28725 in panel 1938 recorded 276 and 123 cycles respectively. These are relatively high numbers and might indicate a possible case o
    f short cycling. [Visualization URL for 28726](https://app.verdigris.co/data/org/8/building/530/panel/1938/breaker/28726?timespanstart=2024-10-02T00%3A00%
    3A00Z&timespanend=2024-10-03T00%3A00%3A00Z&timeZone=UTC&resolution=1m&view=sum&timezonemode=UTC) and [for 28725](https://app.verdigris.co/data/org/8/build
    ing/530/panel/1938/breaker/28725?timespanstart=2024-10-02T00%3A00%3A00Z&timespanend=2024-10-03T00%3A00%3A00Z&timeZone=UTC&resolution=1m&view=sum&timezonem
    ode=UTC)
    
    3. Breaker 29051 in panel 1955 recorded only 4 cycles which is within the normal range.
    
    4. Breaker 29050 in panel 1955 recorded 244 cycles which is considered high and may imply short cycling. [Visualization URL](https://app.verdigris.co/data
    /org/8/building/530/panel/1955/breaker/29050?timespanstart=2024-10-02T00%3A00%3A00Z&timespanend=2024-10-03T00%3A00%3A00Z&timeZone=UTC&resolution=1m&view=s
    um&timezonemode=UTC)
    
    5. A significant number of breakers in various panels (1936, 1937, 1938, 1939, 1941, 1942, 1943, 1944, 1945, and 1955) recorded 0 cycles. This suggests th
    at these breakers were either not in use or are potentially malfunctioning.
    
    Recommendations:
    
    1. For breakers showing many cycles, like 28726, 28725, and 29050, it is crucial to investigate further as this might be a sign of short cycling. Short cy
    cling can lead to increased energy consumption and potential damage to the equipment.
    
    2. For breakers with 0 cycles, ensure that they are functioning properly. If they are meant to be in operation and are showing 0 cycles, it might indicate
     a problem that needs immediate attention.
    
    3. Regularly monitor the cycles of all breakers, even those with moderate or low cycles, to ensure optimal energy usage and prevent any potential issues.
    
    4. Implement a predictive maintenance strategy based on the cycle data to identify potential issues before they lead to equipment failure or increased ene
    rgy costs.
    
    """
]

# 텍스트와 함께 벡터 저장
for i, text in enumerate(texts):
    embedding = get_embedding(text)  # 텍스트를 임베딩으로 변환
    index.upsert([(f"id-{i}", embedding, {"text": text})])  # 메타데이터로 텍스트 정보 추가
