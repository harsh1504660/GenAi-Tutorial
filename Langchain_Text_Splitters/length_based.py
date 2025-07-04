from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

#text ='A paragraph is a distinct unit of writing, typically composed of multiple sentences, that focuses on a single idea or topic. It serves as a building block for longer pieces of text, helping to organize and structure information for clarity and readability.'

loader = PyPDFLoader(file_path='dl-curriculum.pdf')

docs = loader.load()



splitter = CharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result)