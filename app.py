from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from chromadb import PersistentClient

import tempfile
import traceback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Path to persist ChromaDB
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_store"

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Vector Store API. Use /docs to explore endpoints."}


@app.post("/embed_pdf/")
async def embed_pdf(file: UploadFile = File(...)):
    file_path = ""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            file_path = tmp.name

        # Load and split the PDF content
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No readable content found in the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Light embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # ChromaDB client and vector store
        client = PersistentClient(path=CHROMA_DIR)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            client=client
        )

        return JSONResponse(content={"message": f"{len(splits)} chunks embedded and stored successfully."})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.post("/ask_question/")
async def ask_question(question: str = Form(...)):
    try:
        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not found in environment.")

        # Light embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Load ChromaDB with new client
        client = PersistentClient(path=CHROMA_DIR)
        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(question)

        if not retrieved_docs:
            return JSONResponse(content={"answer": "I'm not aware of that. It wasn't in the documents."})

        # Call Claude model
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY)

        system_prompt = (
            "You are an assistant for answering questions based on document context. "
            "Use the provided context below to answer the question. "
            "If the answer is not contained in the context, say: "
            "'I'm not aware of that. It wasn't in the documents.'\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        chain = create_stuff_documents_chain(llm, prompt)

        result = chain.invoke({
            "input": question,
            "context": retrieved_docs
        })

        return JSONResponse(content={"answer": result})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
