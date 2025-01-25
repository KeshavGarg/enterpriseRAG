# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PythonLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embed import embed
# from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

import os
import shutil
import argparse

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()
    else:
        documents = load()
        chunks = split(documents)
        add_db(chunks)

# def load():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()

def load():
    loaders = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loaders.append(PyPDFLoader(file_path))
            elif file.endswith(".txt"):
                loaders.append(TextLoader(file_path))
            elif file.endswith(".py"):
                loaders.append(PythonLoader(file_path))
            elif file.endswith(".md"):
                loaders.append(UnstructuredMarkdownLoader(file_path))
            elif file.endswith(".docx"):
                loaders.append(UnstructuredWordDocumentLoader(file_path))
            else:
                print(f"Unsupported file format: {file}")
    
    # Load all documents
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    return documents

def split(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
    return text_splitter.split(documents)

def add_db(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed())
    
    chunks_with_ids = clac_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    print(f"Current documents in the database: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks) > 0:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("No new documents to add")

def clac_ids(chunks):
    
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    
    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()