import torch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

try:
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print("Embeddings loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
