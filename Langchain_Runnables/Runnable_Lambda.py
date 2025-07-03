from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)


joke_gen_chain = RunnableSequence(prompt,model,parser)


def word_count(text):
    return (len(text.split()))
parllel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})


final_chain =RunnableSequence(joke_gen_chain,parllel_chain)

print(final_chain.invoke({'topic':'python'}))