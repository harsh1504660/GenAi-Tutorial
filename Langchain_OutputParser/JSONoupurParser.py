from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# in the prompt template, we can use the format instruction from the parser
# to ensure the output is in the correct format
template1 = PromptTemplate(
    template='give me a name,age and a city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions() }
)

prompt = template1.format()

result = model.invoke(prompt)

final = parser.parse(result.content)
print(final)