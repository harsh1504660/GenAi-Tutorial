from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)


chat_history = [
    SystemMessage(content="You are a helpful ai assistant"),
]

while True:
    user_input = input("You :")
    chat_history.append(HumanMessage(content=user_input))
    if user_input =='exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content =result.content))
    print("AI : ", result.content)

print("Chat History:", chat_history)