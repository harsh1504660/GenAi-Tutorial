from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # You can change this to any other model available on Hugging Face
)

video_id = "Gfr50f6ZBvo"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])

    # flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    #print(transcript)
except TranscriptsDisabled:
    print("No caption avaible")


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])

vector_store = FAISS.from_documents(chunks,embedding)

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


parser = StrOutputParser()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template="""
    You are a helpful assistant,
    Answer ONLY from provided transcript context.
    if the context is insufficient just say you dont know,
    \n
    context : {context} \n
    Question : {question}
""",
input_variables=["context","question"]

)


main_chain = parallel_chain | prompt | model | parser

print(main_chain.invoke('Can you summarize the video in short'))