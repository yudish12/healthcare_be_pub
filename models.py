from typing import List, Optional
from pydantic import BaseModel


class Message (BaseModel):
    query: str
    answer: str
    thread_id: str
    user_id: str
