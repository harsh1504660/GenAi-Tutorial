from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict

load_dotenv()



#schema
class Review(TypedDict):
    summary: str
    sentiment: str

model = ChatOpenAI()

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("the hardware is good but the software is not that great there are some bugs in the software but overall it is a good product. preinstalled app that i cant uninstall is a bloatware. the hardware is good but the software is not that great there are some bugs in the software but overall it is a good product. preinstalled app that i cant uninstall is a bloatware.")

print(result)