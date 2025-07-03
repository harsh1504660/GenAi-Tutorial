from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader


loader = DirectoryLoader(
    path ='books',
    glob='*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(len(docs))   ### 1 page 1 document

#print(docs[15].page_content)

docs = loader.lazy_load()
for document in docs:
    print(document.metadata)