# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:25:28 2025

@author: z048540
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
import pickle
import pdfplumber
from langchain.schema import Document
from tqdm.auto import tqdm
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

pdf_paths = ['book_A_Text_Book_of_Food_and_Nutrition.pdf']
data = []

for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    document = Document(page_content=text)
    data.append(document)
    
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs1 = text_splitter.split_documents(data)

'''with open('test.pkl', 'wb') as f:
    pickle.dump(docs, f)
    
print("docs stored as pickle file")'''

from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
faiss_vectorstore = FAISS.from_documents(docs1, embeddings)
faiss_vectorstore.save_local("faiss_index")

print("vector embeddings saved")
