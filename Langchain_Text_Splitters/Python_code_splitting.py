from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
    class NakliLLM(Runnable) :
    def __init__(self):
        print("llm created")

    def invoke(self, prompt):
        response_list = [
            'delhi is the capital of India',
            'ipl is the best cricket league in the world',
            'India won the 2023 world cup',
            'India is the best country in the world',
        ]

        return {'response':random.choice(response_list)}
    

    # def predict(slef,prompt):
    #     response_list = [
    #         'delhi is the capital of India',
    #         'ipl is the best cricket league in the world',
    #         'India won the 2023 world cup',
    #         'India is the best country in the world',
    #     ]

    #     return {'response':random.choice(response_list)}
    

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(chunks[1])

