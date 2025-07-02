from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)



class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age:int = Field(gt=18,description="Age of the person")
    city:str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate a fictional {place} person.\n"
        "ONLY return the output in JSON format. No code, no explanation.\n"
        "{format_instructions}"
    ),
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


prompt = template.invoke({'place':'Indian'})

result = model.invoke(prompt)

final = parser.parse(result.content)

print(final)