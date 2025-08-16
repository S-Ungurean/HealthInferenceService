from pydantic import BaseModel

# ---- Pydantic input ----
class PredictInputRequest(BaseModel):
    modelType: str
    content: str  # base64-encoded image string