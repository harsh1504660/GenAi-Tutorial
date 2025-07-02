from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small', # 'text-embedding-3-small' is the latest model as of October 2023
    dimensions=32

)

result = embeddings.embed_query("What is the capital of India?")
# Output: [-0.002, 0.001, -0.003, ...
print(str(result))

