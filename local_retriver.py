from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# -------------------- Embeddings --------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# -------------------- LLM (Phi-3) --------------------
model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    return_full_text=False
)

local_llm = HuggingFacePipeline(pipeline=generator)

# -------------------- PDF Loading --------------------
loader = PyPDFLoader("ML.pdf")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(raw_docs)

# -------------------- ChromaDB Setup --------------------
persist_dir = "local_rag_db"

if not os.path.exists(persist_dir):
    print("Creating new ChromaDB...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir,
        collection_name="rag_chunks"
    )
    vectordb.persist()
else:
    print("Loading existing ChromaDB...")

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name="rag_chunks"
    )

# -------------------- Retriever --------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

question = input("Enter the question: ")

results = retriever.invoke(question)
context = "\n\n".join([d.page_content for d in results])

# -------------------- Prompt --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly knowledgeable ML assistant. "
     "Use ONLY the following context to answer.\n\n"
     "Context:\n{context}\n\n"),
    ("human", "{question}")
])

# -------------------- Final Chain --------------------
chain = prompt | local_llm

response = chain.invoke({
    "context": context,
    "question": question
})

print(response)
