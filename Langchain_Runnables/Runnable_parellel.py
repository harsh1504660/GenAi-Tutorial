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

prompt1 = PromptTemplate(
    template="generate a tweet about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a linkedin post about {topic}",
    input_variables=["topic"],  
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {
        'tweet':RunnableSequence(prompt1,model,parser),
        'linkedin_post':RunnableSequence(prompt2,model,parser)
    }
)


result = parallel_chain.invoke({"topic":"python"})


print(result)