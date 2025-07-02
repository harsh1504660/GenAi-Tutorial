from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # You can change this to any other model available on Hugging Face
)

text="delhi is the capital of india"

embedding_vector = embedding.embed_query(text)

print(str(embedding_vector))