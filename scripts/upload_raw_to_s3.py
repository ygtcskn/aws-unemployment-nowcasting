from pathlib import Path

import boto3

sts = boto3.client("sts")

identity = sts.get_caller_identity()

print("AWS connection successful")
print('ARN:', identity["Arn"])

# AWS configuration
BUCKET_NAME = "aws-unemp-nowcast-ygtcskn"
REGION = "eu-central-1"

s3 = boto3.resource('s3', region_name=REGION)

s3.create_bucket(Bucket=BUCKET_NAME,
                 CreateBucketConfiguration={'LocationConstraint': REGION}
                 )

print(f"Bucket {BUCKET_NAME} created")

# Local data
LOCAL_DIR = Path("data/raw/google_trends")

# S3 destination
S3_PREFIX = "raw/google_trends"

# Connect to S3
s3 = boto3.client("s3", region_name=REGION)

# Find all CSV files
csv_files = sorted(LOCAL_DIR.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files.")

# Upload each file to S3

for i, file_path in enumerate(csv_files, start=1):

    relative_path = file_path.relative_to(LOCAL_DIR).as_posix()
    s3_key = f"{S3_PREFIX}/{relative_path}"

    print(f"[{i}/{len(csv_files)}] Uploading {relative_path}")

    s3.upload_file(
        str(file_path),
        BUCKET_NAME,
        s3_key
    )

print("Upload complete!")
