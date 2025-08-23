import json
import os
import tempfile
import boto3

# Create a top-level session using the s3uploader profile
s3 = boto3.client("s3", region_name="us-east-1")

metadata = {}

# ---- Model loader from S3 ----
def loadModelFromS3(bucket: str, key: str):
    # Get a proper temp file path
    localPath = os.path.join(tempfile.gettempdir(), os.path.basename(key))

    # Always download and overwrite
    s3.download_file(bucket, key, localPath)

    return localPath


# ---- Model Metadata loader from S3 ----
def loadModelMetadatFromS3(bucket: str, key: str):
    # Get a proper temp file path
    localPath = os.path.join(tempfile.gettempdir(), os.path.basename(key))

    # Always download and overwrite
    s3.download_file(bucket, key, localPath)


    with open(localPath, "r") as f:
        content = f.read()
        metadata = json.loads(content)

    return metadata