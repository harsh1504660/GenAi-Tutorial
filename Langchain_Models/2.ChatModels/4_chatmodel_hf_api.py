from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
print("jay")
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

print("here")


model = ChatHuggingFace(llm=llm)

print("or mayve here")

result = model.invoke("What is the capital of India")

print(result.content)