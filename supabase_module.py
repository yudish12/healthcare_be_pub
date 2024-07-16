import re
import time
from datetime import datetime
from supabase import create_client, Client
from supabase.client import ClientOptions
from models import Message


SUPABASE_URL = "https://ujddyzyzyytvhfstxmpr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVqZGR5enl6eXl0dmhmc3R4bXByIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjA3MDM4ODYsImV4cCI6MjAzNjI3OTg4Nn0.397F4KJ4g_dF_9JvoiKfIbdsg35FTVcs757HqUK-F9g"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_chat_history(thread_id: str):
    try:
        response = supabase.table("conversations").select(
            "*").eq("thread_id", thread_id).order("created_at", desc=True).limit(2).execute()
        if response.data:
            # Flatten the response data into the desired format
            chat_history = []
            for row in response.data:
                chat_history.append(row["query"])
                chat_history.append(row["answer"])
            return chat_history
        else:
            return []
    except Exception as e:
        print(e)
        return e


def insert_message(message: Message):
    try:
        response = supabase.table("conversations").insert(
            message.dict()).execute()
        if response.data:
            return response.data
        else:
            raise Exception("Something went wrong")
    except Exception as e:
        print(e)
        return e
