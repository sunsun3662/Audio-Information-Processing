import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# Download BERT models from HF mirror
models = [
    'bert-base-multilingual-uncased',
    'bert-base-uncased',
]

for model_id in models:
    print(f"Downloading {model_id}...")
    try:
        snapshot_download(model_id, resume_download=True)
        print(f"  {model_id} OK")
    except Exception as e:
        print(f"  {model_id} failed: {e}")

print("Done")
