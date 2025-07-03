from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template="generate 5 intresting facts about {topic} ",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = prompt | model | parser

#result = chain.invoke({'topic':'cricket'})

#print(result)

chain.get_graph().print_ascii()