from fastapi import HTTPException
from models import PredictInputRequest


def validateInputRequest(request: PredictInputRequest):
    if not request.modelName:
        raise HTTPException(status_code=400, detail="modelName is required")
    if not request.content:
        raise HTTPException(status_code=400, detail="content is required")