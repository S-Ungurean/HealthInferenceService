from pydantic import BaseModel

# ---- Pydantic input ----
class PredictInputRequest(BaseModel):
    modelName: str
    content: str  # base64-encoded image string