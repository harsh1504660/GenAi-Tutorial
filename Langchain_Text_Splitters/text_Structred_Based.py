from langchain.text_splitter import RecursiveCharacterTextSplitter

text = '''The Elements of Typographic Style states that "at least one en [space]" should be used to indent paragraphs after the first,[2] noting that that is the "practical minimum".[3] An em space is the most commonly used paragraph indent.[3] Miles Tinker, in his book Legibility of Print, concluded that indenting the first line of paragraphs increases readability by 7%, on average.[4]

When referencing a paragraph, typographic symbol U+00A7 § SECTION SIGN (&sect;) may be used: "See § Background".

In modern usage, paragraph initiation is typically indicated by one or more of a preceding blank line, indentation, an "Initial" ("drop cap") or other indication. Historically, the pilcrow symbol (¶) was used in Latin and western European languages. Other languages have their own marks with similar function.'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size =100,
    chunk_overlap =0
)

chunks = splitter.split_text(text)


print(len(chunks))
print(chunks)