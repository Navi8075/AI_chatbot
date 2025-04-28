# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:38:38 2025

@author: z048540
"""

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pickle
from dotenv import load_dotenv
from faster_whisper import WhisperModel



def chatbot_response(audio):
    load_dotenv()
    
    query =''
    model_size = 'base'
    model = WhisperModel(model_size,device='cpu',compute_type='float32')
    segments, info = model.transcribe(audio)
    
    for segment in segments:
        query=segment.text
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_vectorstore_loaded = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    retriever = faiss_vectorstore_loaded.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500)
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. keep the answer in details exactly same as the context."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )   
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return(query,response["answer"])