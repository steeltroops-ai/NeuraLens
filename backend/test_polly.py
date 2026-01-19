"""Test Amazon Polly integration"""
import os
from dotenv import load_dotenv

load_dotenv()

import boto3
from botocore.exceptions import ClientError

aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'ap-south-1')

print(f"AWS Access Key: {aws_key[:15]}..." if aws_key else "AWS_ACCESS_KEY_ID not set!")
print(f"AWS Secret Key: {aws_secret[:5]}..." if aws_secret else "AWS_SECRET_ACCESS_KEY not set!")
print(f"AWS Region: {aws_region}")

try:
    client = boto3.client(
        'polly',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region
    )
    
    # Test with standard engine first (more widely available)
    print("\nTesting Polly with standard engine...")
    response = client.synthesize_speech(
        Engine='standard',
        LanguageCode='en-US',
        OutputFormat='mp3',
        Text='Hello from Amazon Polly',
        VoiceId='Joanna'
    )
    
    audio_bytes = response['AudioStream'].read()
    print(f"SUCCESS! Generated {len(audio_bytes)} bytes of audio")
    
    # Save to file for verification
    with open("test_polly.mp3", "wb") as f:
        f.write(audio_bytes)
    print("Saved audio to test_polly.mp3")
    
except ClientError as e:
    error_code = e.response['Error']['Code']
    error_msg = e.response['Error']['Message']
    print(f"\nAWS Error: {error_code}")
    print(f"Message: {error_msg}")
    
    if "AccessDenied" in str(e) or "not authorized" in str(e).lower():
        print("\n>>> The IAM user does not have Polly permissions!")
        print(">>> Go to AWS IAM Console and attach 'AmazonPollyFullAccess' policy to your user.")
        
except Exception as e:
    print(f"\nError: {e}")
