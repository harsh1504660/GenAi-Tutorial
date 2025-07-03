from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="expplain this joke \n {joke}",
    input_variables=["joke"],
)


joke_gen_chain = RunnableSequence(prompt1, model, parser)

parllel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain, parllel_chain)

print(final_chain.invoke({"topic": "python"}))