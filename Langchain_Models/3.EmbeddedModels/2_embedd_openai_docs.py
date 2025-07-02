from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()
documents = [    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small', # 'text-embedding-3-small' is the latest model as of October 2023
    dimensions=32

)

result = embeddings.embed_documents(documents)
# Output: [-0.002, 0.001, -0.003, ...
print(str(result))

