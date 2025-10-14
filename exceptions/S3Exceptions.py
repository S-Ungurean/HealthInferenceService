class ModelLoadError(Exception):
    """Base exception for model loading failures."""
    pass

class S3DownloadError(ModelLoadError):
    """Raised when an S3 download fails."""
    def __init__(self, bucket: str, key: str, original_exception: Exception):
        self.bucket = bucket
        self.key = key
        self.original_exception = original_exception
        msg = f"Failed to download file from S3 (bucket={bucket}, key={key}): {original_exception}"
        super().__init__(msg)

class MetadataParseError(ModelLoadError):
    """Raised when model metadata JSON fails to parse."""
    def __init__(self, path: str, original_exception: Exception):
        self.path = path
        self.original_exception = original_exception
        msg = f"Failed to parse model metadata JSON from file {path}: {original_exception}"
        super().__init__(msg)