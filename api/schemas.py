from pydantic import BaseModel
from typing import List
from datetime import datetime

class Transaction(BaseModel):
    transaction_id: str
    timestamp: datetime
    amount: float
    transaction_type: str  # 'bet' or 'deposit'

class PredictPayload(BaseModel):
    user_id: int
    transactions: List[Transaction]

class Token(BaseModel):
    access_token: str
    token_type: str
