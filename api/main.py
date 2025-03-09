from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Union
import logging
import psutil
import os
import urllib.parse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# FastAPI app initialization
app = FastAPI()

class RAGPipeline:
    def __init__(self, model_name: str = "llama2:7b-chat-q4", max_memory_gb: float = 3.0):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        
        # Load the language model (LLM)
        #self.llm = OllamaLLM(model="deepseek-r1:8b")
        
        self.llm= ChatGroq(model="deepseek-r1-distill-qwen-32b",
                           api_key= "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",
                        )


        # Initialize embeddings using a lightweight model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for efficiency
        )
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template(""" 
        Answer the question based only on the following context.If you cannot answer the question on the following content then answer with your normal intellegence.
        If you cannot find the answer in the context, say "I cannot answer this based on the provided context,and give your general answers as this is what I know"
        
        Context: {context}
        Question: {question}
        Answer: """)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")

    def load_and_split_documents(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(splits)} document chunks")
        return splits

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        batch_size = 32
        vectorstore = FAISS.from_documents(documents[:batch_size], self.embeddings)
        
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i // batch_size + 1}")
        return vectorstore

    def setup_rag_chain(self, vectorstore: FAISS):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2, "fetch_k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough() }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain


    def save_embeddings_chroma(self, documents: List[Document], persist_directory: str = "./chroma_db"):
        """Save embeddings into ChromaDB"""
        vectorstore = Chroma.from_documents(documents, self.embeddings, persist_directory=persist_directory)
        vectorstore.persist()
        self.logger.info(f"Embeddings saved to {persist_directory}")
        return vectorstore

    
    def query(self, chain, question: str) -> str:
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        return chain.invoke(question)


# Initialize the RAGPipeline once
rag = RAGPipeline(model_name="deepseek-r1-distill-qwen-32b", max_memory_gb=3.0)


@app.get("/")
def read_root():
    # Return a simple message indicating the API is working
    return {"message": "RAGPipeline FastAPI Integration is working!"}


@app.post("/rag/query/{question}")
async def rag_query(question: str, file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        file_location = f"./temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        
        # Load and process the document
        documents = rag.load_and_split_documents(file_location)
        vectorstore = rag.create_vectorstore(documents)
        chain = rag.setup_rag_chain(vectorstore)

        # Query the RAG pipeline
        #prePrompt = """You are a helpful assistant. You will be provided with a context and a question. Your task is to answer the question based on the context provided. or if you dont have context just answer based on your general intelligence. Question:"""
        ques=   question 
        response = rag.query(chain, ques)
        
        # Return the answer to the question
        return {"question": question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/embed")
async def embed_documents(file: UploadFile = File(...)):
    try:
        file_location = f"./temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        
        # Load and process the document
        documents = rag.load_and_split_documents(file_location)
        vectorstore = rag.save_embeddings_chroma(documents)
        
        return {"message": "Embeddings stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/del")
async def delete_embeddings():
    try:
        success = rag.delete_embeddings_chroma()
        if success:
            return {"message": "Embeddings deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete embeddings")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# To run the FastAPI app, use: uvicorn main:app --reload
