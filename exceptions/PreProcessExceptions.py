class PreprocessError(Exception):
    """Base exception for preprocessing failures."""
    pass

class UnsupportedInputTypeError(PreprocessError):
    """Raised when input_type in metadata is not supported."""
    def __init__(self, input_type: str):
        self.input_type = input_type
        msg = f"Unsupported input type: {input_type}"
        super().__init__(msg)

class Base64DecodeError(PreprocessError):
    """Raised when base64 decoding fails."""
    def __init__(self, original_exception: Exception):
        self.original_exception = original_exception
        msg = f"Failed to decode base64 input: {original_exception}"
        super().__init__(msg)