from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
import shutil
import uuid

app = FastAPI()
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


db_chroma = None

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global db_chroma

    # Save uploaded file to a temporary location
    file_id = str(uuid.uuid4())
    temp_path = f"/tmp/{file_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and process PDF
    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        db_chroma = Chroma.from_documents(chunks, embeddings)

        return {"message": "PDF uploaded and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    global db_chroma

    if db_chroma is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a PDF first.")

    try:
        docs_chroma = db_chroma.similarity_search_with_score(question, k=5)
        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        Answer the question based on the above context: {question}.
        Provide a detailed answer.
        Don’t justify your answers.
        Don’t give information not mentioned in the CONTEXT INFORMATION.
        Do not say "according to the context" or "mentioned in the context" or similar.
        """

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)

        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY)
        response_text = llm.predict(prompt)

        return JSONResponse(content={"answer": response_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
