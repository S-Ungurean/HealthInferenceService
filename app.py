from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import torch
import torchvision.models as models
import torch.nn as nn

from accessor.S3Accessor import loadModelFromS3, loadModelMetadataFromS3
from accessor.YAMLAccessor import getModelInfo, loadConfig
from configs import ModelsMap
from exceptions.AccessorExceptions import ConfigError
from exceptions.PreProcessExceptions import PreprocessError
from exceptions.S3Exceptions import ModelLoadError
from models.PredictInputRequest import PredictInputRequest
from transformer.InputTransformer import preprocess
from logger import get_logger
from validators.APIInputValidators import validateInputRequest

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

    validateInputRequest(request)

    logger.info("Received /predict request for modelType=%s", request.modelName)

    modelName = request.modelName
    content = request.content

    # Load YAML Config
    try:
        yamlConfig = loadConfig(str(YAML_RELATIVE_PATH))
        modelInfo = getModelInfo(yamlConfig, modelName)
    except ConfigError as e:
        logger.error("Config loading failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


    modelName = modelInfo['model_name']
    modelMetadataName = modelInfo['model_metadata']
    logger.info("Resolved modelName=%s, modelMetadataName=%s", modelName, modelMetadataName)

    # Load model and metadata from s3
    try:
        modelMetadata = loadModelMetadataFromS3(S3_BUCKET_NAME, modelMetadataName)
        modelPath = loadModelFromS3(S3_BUCKET_NAME, modelName)
    except ModelLoadError as e:
        logger.error("Model loading failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    outputClasses = modelMetadata["output"]

    # Load models into workspace
    try:
        architecture = modelMetadata["arch"]
        model = initializeModelArchitecture(architecture, len(outputClasses))
        model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
        model.eval()

        # Preprocess
        logger.info("Preprocessing input data")
        inputTensor = preprocess(content, modelMetadata)

        # Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            outputs = model(inputTensor.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        predicted_class = outputClasses[pred.item()]
        logger.info("Inference completed. Predicted class=%s, confidence=%.4f", predicted_class, float(conf.item()))
    except (Exception, PreprocessError)  as e:
        logger.exception("Error during model inference")
        raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": float(conf.item())
    })

def initializeModelArchitecture(architecture, num_classes):
    logger.info("Initializing model architecture: %s", architecture)
    if architecture not in ModelsMap.model_map:
        logger.error("Unknown architecture type: %s", architecture)
        raise ValueError(f"Unknown architecture type: {architecture}")

    model = ModelsMap.model_map[architecture](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model