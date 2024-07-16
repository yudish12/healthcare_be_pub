from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from utils import *
from supabase_module import get_chat_history, insert_message
from models import Message

app = FastAPI()

origins = [
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes


@app.get("/")
def index():
    return "Welcome to MEDIBOT !!"


@app.post("/generate_response_open_bio_llm_llama3/")
def response_1(query: str, chat_history: str):
    biollm_response = generate_answer(query)
    encoded_biollm_response = model.encode(str(biollm_response))
    vector_ids = get_vector_ids(encoded_biollm_response)

    chunks = fetch_text_chunks(vector_ids)
    response = groq_llama(str(biollm_response), str(chunks), chat_history)

    return response


@app.post("/generate_response_open_bio_llm_only/")
def response_2(query: str, chat_history: str):

    encoded_query = model.encode(str(query))
    vector_ids = get_vector_ids(encoded_query)

    chunks = fetch_text_chunks(vector_ids)
    response = generate_answer_biollm_only(query, str(chunks), chat_history)
    return response


@app.post("/generate_response_med42_llama3/")
def response_3(query: str, thread_id: str):
    chat_history = get_chat_history(thread_id)
    refined_query = generate_query(query, chat_history)
    med42_response = generate_answer_med_42(refined_query)
    encoded_med42_response = model.encode(str(med42_response))
    vector_ids = get_vector_ids(encoded_med42_response)

    chunks = fetch_text_chunks(vector_ids)
    total_chunks_unique=[]
    for item in chunks:
        total_chunks_unique.append(item[1])
    total_chunks_unique=set(total_chunks_unique)
    response = groq_llama(str(med42_response), str(total_chunks_unique))

    message = Message(
        query=query,
        answer=response,
        thread_id=thread_id,
        user_id="4bd6600b-c4ee-4272-b891-1e20aeb4ac7a"
    )
    insert_message(message)

    return response


@app.post("/generate_response_med42_only/")
def response_4(query: str, thread_id: str):
    chat_history = get_chat_history(thread_id)

    refined_query = generate_query(query, chat_history)
    encoded_query = model.encode(refined_query)
    vector_ids = get_vector_ids(encoded_query)
    chunks = fetch_text_chunks(vector_ids)
    total_chunks_unique=[]
    for item in chunks:
        total_chunks_unique.append(item[1])
    total_chunks_unique=set(total_chunks_unique)
    response = generate_answer_med_42_only(query, str(total_chunks_unique))

    message = Message(
        query=query,
        answer=response,
        thread_id=thread_id,
        user_id="4bd6600b-c4ee-4272-b891-1e20aeb4ac7a"
    )
    insert_message(message)

    return response
