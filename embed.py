# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

def embed():
    # embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")

    # all local
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings