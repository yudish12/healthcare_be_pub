
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SpacyTextSplitter
import requests
import json
import psycopg2
from pinecone.grpc import PineconeGRPC as Pinecone
from groq import Groq
from openai import OpenAI
import os

# EMBEDDING MODEL

model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

# PINECONE

api_key = 'cc42103e-d7b7-445a-be3e-8821cbc67293'
pc = Pinecone(api_key=api_key)


def get_vector_ids(embedding):
    index = pc.Index("healthcarepubmed")

    response=index.query(
        vector=embedding,
        top_k=8,
        include_values=False,
        include_metadata=False
    )
    # parsing the ids in responses 
    ids=[]
    for item in response["matches"]:
        ids.append(item["id"])

    return ids



# POSTGRESQL
def fetch_text_chunks(id_list):
    id_list=tuple(id_list)
    connection = psycopg2.connect(
            dbname='railway',
            user='postgres',
            password='fSXBHrRYpyUDTtERiDDZoKTqcsnQKQzM',
            host='roundhouse.proxy.rlwy.net',
            port=25614
        )
    cursor = connection.cursor()
    cursor.execute(f"SELECT id, text_chunks FROM healthcare2 WHERE id IN {id_list}")
    text_chunk = cursor.fetchall()
    cursor.close()
    connection.close()

    return text_chunk


# OPEN BIO LLM

API_URL = "https://gs12wjfkr4isbwip.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json"
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def generate_answer(question):
    try:
        prompt = f"""
        Question: {question}
        Answer:"""

        output = query({
            "inputs": prompt,
            "parameters": {
                "temperature": 0.01
            }
        })

        return output[0]['generated_text']
    except Exception as e:
        print("Error occured due to : ", e)

def generate_answer_biollm_only(query,chunks,chat_history=None):
    try:
        prompt = f"""
        You are a medical expert who can provide  accurate consultations given informaiton about any person's health ocndiitons and medical history. You will be given a question , and then also some supporting or enhancing information which should  be related to the question. Figure out correct anser using all the information provided to you. You will also be provided with a chat history and maintain coherent conversations with continuity.
        \n\nQuestion: {query}
        \n\nInformation:{chunks}

        \n\nChat_history: {chat_history}
        \n\nAnswer:"""

        output = query({
            "inputs": prompt,
            "parameters": {
                "temperature": 0.01
            }
        })

        return output[0]['generated_text']
    except Exception as e:
        print("Error occured due to : ", e)



# MISTRAL RESPONSE

MISTRAL_URL = 'https://Mistral-large-deploy-serverless.eastus2.inference.ai.azure.com/v1/chat/completions'
MISTRAL_API_KEY = 'RlbBY2ya6JffeQCdABi696QT1R1X0eDd'
s = requests.Session()
headers = {"Content-Type": "application/json", "Authorization": (MISTRAL_API_KEY)}


def mistral_response(discussion,chunks,chat_history=None):


    data = {
    "messages": [
        {"role": "system", "content": "You are a helpful medical assistant.You are required to understand a brief medical Q&A, infer more relevant information from the chunks provided and improve or support the answer only with meaningful content. You will not invent any information on your own and will not speak outside of the context provided. you are required to strictly output responses in Markdown format.Consider the discussion to be factually true and include it's diagnosis and treatment in the final answer."},
        {"role": "user", "content": f"The Q&A discussion is as {discussion}.\n\n The additional chunks for supporting/improving the answer are :{chunks}.\n\n The chat history is: {chat_history}"},

    ],
    "max_tokens": 4000,
    "temperature": 0.01,
    }

    response=''
    with s.post(MISTRAL_URL, data=json.dumps(data), headers=headers, stream=True) as resp:
        print(resp.status_code)
        for line in resp.iter_lines():
            response+=line.decode('utf8', 'ignore')

    result = json.loads(response)
    return result["choices"][0]["message"]['content']

# GROQ LLAMA #

def groq_llama(discussion,chunks):
    client = Groq(
        api_key="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",
    )

    chat_completion = client.chat.completions.create(
        messages = [{
            "role": "user",
            "content": f'''You have been provided a discussion and chunks.
            You need to improve the answer in discussion based on the chunks provided if possible. Always be conversational,try mentioning the given information first and use words like "As ..", or "Based on the information provided" and then move ahead with the answer .Only if required, create a well structured headings/subheadings summary in Markdown format for diagnosis, treatment, management, cure etc (use relevant headings case to case). Respond in Markdown format. Please understand any medical and healthcare-related question and answer it in detail.
            
            The Q&A is: {discussion}.
            The additional chunks to improve the answer is : {chunks}.
            If those additional chunks donot help in answering the patient questions, donot use them in generating your response. And donot mention anything about "additional chunks" in your response if they are not useful.
            Your response should never be solely based on additional chunks provided, those are to add more details to your answer if they are relevant.
            Do not respond to jokes, or questions related to political, demographic, or social issues. Simply state that you are not permitted to answer anything except core medical queries.
            If you don't know the answer to a question, simply state, "I don't know the answer, can I help you with something else?"

            Attach the summary only with relevant information in Markdown format at the end of the paragraph

            Your answer should be a single, well-structured paragraph without repeating the question or mentioning these instructions. Do not use phrases like 'as a medical assistant'. Provide a clear, concise, and professional response in a doctor's tone.''',}],
        model="llama3-8b-8192",
        stream=True,
        max_tokens=4000,
        temperature=0.01,
    )

    chunks=''
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            chunks+=chunk.choices[0].delta.content

    return chunks





def generate_answer_med_42(query):

    print("query: ",query)
    endpoint_url ="https://api.runpod.ai/v2/vllm-dd9f40y17htzpc/openai/v1"
    api_key = "8PG087BNQAVLA5DCGLEFN35QV3Q16IW18OX1DO6A"

    client = OpenAI(
        base_url=endpoint_url,
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        model="m42-health/Llama3-Med42-8B",
        messages = [{
            "role": "user",
            "content": f'''Answer the following medical and healthcare-related question in detail.
            Base your response primarily on your medical knowledge.Always be conversational,try mentioning the given information first and use words like "As ..", or "Based on the information provided" and then move ahead with the answer .Only if required, create a well structured headings/subheadings summary in Markdown format for diagnosis, treatment, management, cure etc (use relevant headings case to case). Respond in Markdown format. Please understand any medical and healthcare-related question and answer it in detail.
            
            The question is: {query}.
            
            Do not repeat the question again.
            Do not respond to jokes, or questions related to political, demographic, or social issues. Simply state that you are not permitted to answer anything except core medical queries.
            If you don't know the answer to a question, simply state, "I don't know the answer, can I help you with something else?"

            Attach the summary only with relevant information in Markdown format at the end of the paragraph

            Your answer should be a single, well-structured paragraph without repeating the question or mentioning these instructions. Do not use phrases like 'as a medical assistant'. Provide a clear, concise, and professional response in a doctor's tone.''',}],
            
                max_tokens= 2000,
                temperature=0.01,
                top_p=0.75,

        )

    # print("finishing time: ",time.time()-start)
    return chat_completion.choices[0].message.content


def generate_query(query,chat_history=None):
    endpoint_url ="https://api.runpod.ai/v2/vllm-dd9f40y17htzpc/openai/v1"
    api_key = "8PG087BNQAVLA5DCGLEFN35QV3Q16IW18OX1DO6A"

    client = OpenAI(
        base_url=endpoint_url,
        api_key=api_key,
    )

    import time
    start=time.time()

    chat_completion = client.chat.completions.create(
        model="m42-health/Llama3-Med42-8B",
          messages = [{
            "role":"user",
                "content": f'''"You are a helpful assistant who will refine the medical query based on chat history. Do not answer the medical query, analyze the chat history to see if query can be made better in the context of the chat history. Sometimes patients would ask questions in natural flow of conversation, like summarize this, or explain my disease,or if there is any possibility where user asked something that is in reference to previous user queries then looking at the chat history, you need to add details to the question and make the question easy for the doctor to understand. 
                Just make the query more informative, that's all. Donot answer the query or explain your ratinale for the new query. Donot write or mention chat history anywhere in your answer. Donot make it very lengthy.
                If no chat history is provided, return the same query as is.
                
                \n\n The query is: {query}
                \n\n The chat history is: {chat_history}
                Just give the final refined query."'''}],
            
                max_tokens= 500,
                temperature=0.1,
                top_p=0.3,

    )

    # print(chat_completion.choices[0].message.content)

    # print("finishing time: ",time.time()-start)
    return chat_completion.choices[0].message.content

def generate_answer_med_42_only(query,chunks):
    endpoint_url ="https://api.runpod.ai/v2/vllm-dd9f40y17htzpc/openai/v1"
    api_key = "8PG087BNQAVLA5DCGLEFN35QV3Q16IW18OX1DO6A"

    client = OpenAI(
        base_url=endpoint_url,
        api_key=api_key,
    )

    import time
    start=time.time()

    chat_completion = client.chat.completions.create(
        model="m42-health/Llama3-Med42-8B",
        messages = [{
            "role": "user",
            "content": f'''Answer the following medical and healthcare-related question in detail.
            Base your response primarily on your knowledge, incorporating additional chunks only if they are relevant.Always be conversational,try mentioning the given information first and use words like "As ..", or "Based on the information provided" and then move ahead with the answer   
            Respond in Markdown format. Please understand any medical and healthcare-related question and answer it in detail.Only if required, create a well structured headings/subheadings summary in Markdown format for diagnosis, treatment, management, cure etc (use relevant headings case to case)
            
            The question is: {query}.
            
            The additional chunks (use only if they are relevant to the context): {chunks}.
            If the additional chunks do not help in answering the patient's question, do not use them in your response. Do not mention anything about "additional chunks" in your response if they are not useful.
            
            Your response should never be solely based on the additional chunks provided; those are to add more details to your answer if they are relevant.

            Attach the summary only with relevant information in Markdown format at the end of the paragraph
            Output the response strictly in Markdown format. Do not repeat the question again.
            Do not respond to jokes, or questions related to political, demographic, or social issues. Simply state that you are not permitted to answer anything except core medical queries.
            If you don't know the answer to a question, simply state, "I don't know the answer, can I help you with something else?"

            Your answer should be a single, well-structured paragraph without repeating the question or mentioning these instructions. Do not use phrases like 'as a medical assistant'. Provide a clear, concise, and professional response in a doctor's tone.''',
            }]
            ,
            
                max_tokens= 2000,
                temperature=0.1,
                top_p=0.3,

        )

    # print(chat_completion.choices[0].message.content)

    # print("finishing time: ",time.time()-start)
    return chat_completion.choices[0].message.content
