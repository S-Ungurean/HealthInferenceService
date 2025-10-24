from http.client import HTTPException
import json
import os
import tempfile
import boto3
from exceptions.S3Exceptions import MetadataParseError, S3DownloadError
from logger import get_logger

logger = get_logger(__name__)

# Create a top-level session using the s3uploader profile
s3 = boto3.client("s3", region_name="us-east-1")

metadata = {}

# ---- Model loader from S3 ----
def loadModelFromS3(bucket: str, key: str):
    # Get a proper temp file path
    localPath = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    logger.info("Downloading model file from S3: bucket=%s, key=%s -> %s", bucket, key, localPath)

    try:
    # Always download and overwrite
        s3.download_file(bucket, key, localPath)
        logger.info("Successfully downloaded model to %s", localPath)
    except Exception as e:
        logger.exception("Failed to download model from S3: bucket=%s, key=%s", bucket, key)
        raise S3DownloadError(bucket, key, e)

    return localPath


# ---- Model Metadata loader from S3 ----
def loadModelMetadataFromS3(bucket: str, key: str):
    # Get a proper temp file path
    localPath = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    logger.info("Downloading model metadata from S3: bucket=%s, key=%s -> %s", bucket, key, localPath)


    try:
        # Always download and overwrite
        s3.download_file(bucket, key, localPath)
        logger.info("Successfully downloaded metadata to %s", localPath)
    except Exception as e:
        logger.exception("Failed to download model metadata from S3: bucket=%s, key=%s", bucket, key)
        raise S3DownloadError(bucket, key, e)

    try:
        with open(localPath, "r") as f:
            content = f.read()
            metadata = json.loads(content)
            logger.info("Loaded metadata: %s", metadata)
            return metadata
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse model metadata JSON from file %s", localPath)
        raise MetadataParseError(localPath, e)