import boto3

REGION = "eu-central-1"
BUCKET_NAME = "aws-unemp-nowcast-ygtcskn"

s3 = boto3.resource('s3', region_name=REGION)

s3.create_bucket(Bucket=BUCKET_NAME,
                 CreateBucketConfiguration={'LocationConstraint': REGION}
                 )

print(f"Bucket {BUCKET_NAME} created")