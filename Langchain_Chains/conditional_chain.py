from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)


parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

parser2 =   PydanticOutputParser(pydantic_object=Feedback)    
  


prompt1 = PromptTemplate(
    template="classify the sentiment of the following feddback text into positive or negative \n {feedback} \n {format_instructions}",

    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)   

classfier_chain = prompt1 | model | parser2  

prompt2 = PromptTemplate(
    template= "write an appropitate response to this positve feedback \n {feedback} ",
    input_variables=["feedback"])

prompt3 = PromptTemplate(
    template= "write an appropitate response to this negative feedback \n {feedback} ",
    input_variables=["feedback"])

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',prompt2 | model | parser) ,
    (lambda x:x.sentiment == 'negative',prompt3 | model | parser),
    RunnableLambda(lambda x: "No response needed")  # Default case if no condition matches
)

chain = classfier_chain | branch_chain

print(chain.invoke({'feedback': 'I love the new features of this app!'}))