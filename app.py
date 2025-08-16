from fastapi import FastAPI
from fastapi.responses import JSONResponse

import torch
import torchvision.models as models
import torch.nn as nn

from accessor.S3Accessor import loadModelFromS3, loadModelMetadatFromS3
from accessor.YAMLAccessor import getModelInfo, loadConfig
from configs import ModelsMap
from models.PredictInputRequest import PredictInputRequest
from transformer.InputTransformer import preprocess

app = FastAPI()

# ---- Configuration ----
S3_BUCKET_NAME = "ai-health-model-storage"
YAML_RELATIVE_PATH = "configs\ModelTypes.yaml"
modelName = ""
modelMetadata = ""
architecture = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- API Endpoint ----
@app.post("/predict")
async def predict(request: PredictInputRequest):

    modelType = request.modelType
    content = request.content

    # Load YAML Config
    yamlConfig = loadConfig(YAML_RELATIVE_PATH)
    modelInfo = getModelInfo(yamlConfig, modelType)

    modelName = modelInfo['model_name']
    modelMetadataName = modelInfo['model_metadata']

    # Load model and metadata from s3
    modelMetadata = loadModelMetadatFromS3(S3_BUCKET_NAME, modelMetadataName)
    modelPath = loadModelFromS3(S3_BUCKET_NAME, modelName)

    outputClasses = modelMetadata["output"]

    # Load models into workspace
    architecture = modelMetadata["arch"]
    if architecture not in ModelsMap.model_map:
        raise ValueError(f"Unknown architecture type: {architecture}")

    model = ModelsMap.model_map[architecture](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(outputClasses))
    model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
    model.eval()

    # Preprocess
    inputTensor = preprocess(content)

    # Run inference
    with torch.no_grad():
        outputs = model(inputTensor.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return JSONResponse({
        "predicted_class": outputClasses[pred.item()],
        "confidence": float(conf.item())
    })