from langchain_community.document_loaders import WebBaseLoader

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
    template="answer the following this question {question} on the following text \n {info}",
    input_variables=["question","info"]
)


parser = StrOutputParser()



url = 'https://www.flipkart.com/motorola-edge-60-5g/p/itm18a81b952d716?pid=MOBHB3T9PWMEHZST&param=3635&otracker=clp_bannerads_1_14.bannerAdCard.BANNERADS_Motorola-edge-60-5g-Sale%2BIs%2BOn_mobile-phones-store_A2Y2IMYDNBEQ'
loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'question':'what is the name of the product','info':docs[0].page_content}))