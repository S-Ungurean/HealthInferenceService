from pathlib import Path

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
from logger import get_logger

# ---- Configure Logging ----
logger = get_logger(__name__)

app = FastAPI()

# ---- Configuration ----
S3_BUCKET_NAME = "ai-health-model-storage"
YAML_RELATIVE_PATH = Path("configs") / "ModelTypes.yaml"
modelName = ""
modelMetadata = ""
architecture = ""

device = "cpu" #### HARDCODED CUDA ISSUES
logger.info("Service starting up. Using device %s", device)

# ---- API Endpoint ----
@app.post("/predict")
async def predict(request: PredictInputRequest):
    logger.info("Received /predict request for modelType=%s", request.modelType)

    modelType = request.modelType
    content = request.content

    # Load YAML Config
    yamlConfig = loadConfig(str(YAML_RELATIVE_PATH))
    modelInfo = getModelInfo(yamlConfig, modelType) 

    modelName = modelInfo['model_name']
    modelMetadataName = modelInfo['model_metadata']
    logger.debug("Resolved modelName=%s, modelMetadataName=%s", modelName, modelMetadataName)

    # Load model and metadata from s3
    modelMetadata = loadModelMetadatFromS3(S3_BUCKET_NAME, modelMetadataName)
    modelPath = loadModelFromS3(S3_BUCKET_NAME, modelName)

    outputClasses = modelMetadata["output"]

    # Load models into workspace
    architecture = modelMetadata["arch"]
    if architecture not in ModelsMap.model_map:
        logger.error("Unknown architecture type: %s", architecture)
        raise ValueError(f"Unknown architecture type: {architecture}")

    logger.info("Initializing model architecture: %s", architecture)
    model = ModelsMap.model_map[architecture](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(outputClasses))
    model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
    model.eval()

    # Preprocess
    logger.debug("Preprocessing input data")
    inputTensor = preprocess(content)

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        outputs = model(inputTensor.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = outputClasses[pred.item()]
    logger.info("Inference completed. Predicted class=%s, confidence=%.4f", predicted_class, float(conf.item()))

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": float(conf.item())
    })