class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    pass

class ConfigLoadError(ConfigError):
    """Raised when loading a config file fails."""
    def __init__(self, path: str, original_exception: Exception):
        self.path = path
        self.original_exception = original_exception
        msg = f"Failed to load config from {path}: {original_exception}"
        super().__init__(msg)

class ModelNotFoundError(ConfigError):
    """Raised when a model is not found in the config."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        msg = f"Model name '{model_name}' not found in config."
        super().__init__(msg)