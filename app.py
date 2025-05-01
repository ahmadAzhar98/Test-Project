from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import tempfile
import traceback
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Query API. Visit /docs for Swagger UI."}

@app.post("/query_pdf/")
async def query_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    file_path = ""
    try:
        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not found in environment.")

        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            file_path = tmp.name

        print(f"Saved uploaded PDF to: {file_path}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from PDF.")

        if not docs:
            raise ValueError("No readable content found in the PDF.")

        print("Sample content:", docs[0].page_content[:300])

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks.")

        if not splits:
            raise ValueError("No text chunks were created from the document.")

        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(question)
        print(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        for i, doc in enumerate(retrieved_docs[:2]):
            print(f"Context {i+1}:\n{doc.page_content[:300]}")

        # Setup LLM
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        result = question_answer_chain.invoke({
            "input": question,
            "context": retrieved_docs
        })

        return JSONResponse(content={"answer": result})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Temporary file {file_path} deleted.")
