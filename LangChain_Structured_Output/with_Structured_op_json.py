from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field,Literal
from typing import TypedDict,Annotated

load_dotenv()

 

#schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Key themes or topics discussed in the review"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "description": "The overall sentiment of the review, e.g., positive, negative, neutral"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}
model = ChatOpenAI()

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("the hardware is good but the software is not that great there are some bugs in the software but overall it is a good product. preinstalled app that i cant uninstall is a bloatware. the hardware is good but the software is not that great there are some bugs in the software but overall it is a good product. preinstalled app that i cant uninstall is a bloatware.")

print(result)