import yaml

def loadConfig(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Get model details by type
def getModelInfo(config: dict, model_type: str):
    for model in config.get("models", []):
        if model.get("type") == model_type:
            return model
    raise ValueError(f"Model type '{model_type}' not found in config.")