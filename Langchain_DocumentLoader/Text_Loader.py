from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="write a summery for following poem \n {poem}",
    input_variables=["poem"]
)


parser = StrOutputParser()

loader = TextLoader('cricket.txt',encoding='utf-8')

docs = loader.load()

print(type(docs))   ### list , all are lists

# print(docs[0])         ### page content and meta data

# print(docs[0].metadata)

# print(docs[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))