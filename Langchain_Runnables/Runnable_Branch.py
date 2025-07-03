from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough,RunnableBranch,RunnableSequence
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt1  = PromptTemplate(
    template = "write a detail report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="summarize the following text: \n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

# or (only for runnable seqeuence)
#report_gen_chain = prompt1 | model | parser


branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 50,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain,branch_chain)   

print(final_chain.invoke({"topic":"python"}))