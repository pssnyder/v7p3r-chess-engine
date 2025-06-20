# testing/gcp_credential_test.py
# This script tests Google Cloud credentials and services.
# Ensure you have set the GOOGLE_APPLICATION_CREDENTIALS environment variable correctly.
from google.cloud import storage, firestore
import os
print("Creds:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
# Cloud Storage
client = storage.Client()
print("Buckets:", [b.name for b in client.list_buckets(max_results=3)])
# Firestore
db = firestore.Client()
print("OK with Firestore!")