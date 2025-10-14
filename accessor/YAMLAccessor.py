import yaml

from exceptions.AccessorExceptions import ConfigLoadError, ModelNotFoundError

def loadConfig(config_path: str):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigLoadError(config_path, e)

# Get model details by type
def getModelInfo(config: dict, model_name: str):
    for model in config.get("models", []):
        if model.get("name") == model_name:
            return model
    raise ModelNotFoundError(model_name)