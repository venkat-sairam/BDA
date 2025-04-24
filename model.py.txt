import os
import uuid
import pdf2image
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path, dpi=300):
    pages = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    all_text = ""
    for page_number, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang='eng')
        all_text += f"\n\n--- Page {page_number + 1} ---\n\n" + text
    return all_text

def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return output_path

def load_and_chunk_text(path, chunk_size=500, chunk_overlap=50):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([raw_text])

def create_vector_store(docs, model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(docs, embedding_model)

def retrieve_context(vectorstore, query, k):
    retriever = vectorstore.as_retriever(search_type="similarity", k=k)
    relevant_docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

def build_prompt(context, query):
    return f"""
Here is the article context:

{context}

Now answer the following:

{query}
"""

def generate_answer(prompt, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt)
    return response.text

def run_rag_pipeline(query, document_path, embedding_model_name, top_k):
    docs = load_and_chunk_text(document_path)
    vectorstore = create_vector_store(docs, embedding_model_name)
    context = retrieve_context(vectorstore, query, k=top_k)
    prompt = build_prompt(context, query)
    return generate_answer(prompt)

def ocr_flow(query, pdf_path, output_file):
    text = extract_text_from_pdf(pdf_path)
    save_text_to_file(text, output_file)
    return run_rag_pipeline(query, output_file, "sentence-transformers/all-MiniLM-L6-v2", 5)

def no_ocr_flow(query, pdf_path):
    return run_rag_pipeline(query, pdf_path, "sentence-transformers/all-MiniLM-L6-v2", 5)
