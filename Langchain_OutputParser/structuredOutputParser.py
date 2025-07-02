from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)



schema = [
    ResponseSchema(name='fact_1',description='A 1 fact about the topic'),
    ResponseSchema(name='fact_2',description='A 2 fact about the topic'),
    ResponseSchema(name='fact_3',description='A 3 fact about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'give me 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# chain = template | model | parser
# result = chain.invoke({'topic':'black hole'})
prompt = template.invoke({'topic':'black hole'})

result = model.invoke(prompt)

final = parser.parse(result.content)

print(final)