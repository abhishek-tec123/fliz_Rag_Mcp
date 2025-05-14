import os
from langchain.docstore.document import Document
import tiktoken
import pandas as pd
from io import BytesIO

class FileUploader:
    def __init__(self, model="cl100k_base"):
        self.model = model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def count_tokens(self, text):
        encoder = tiktoken.get_encoding(self.model)
        return len(encoder.encode(text))

    def extract_text_from_upload(self, filename: str, file_obj):
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            return file_obj.read().decode("utf-8")

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(BytesIO(file_obj.read()), sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"--- Sheet: {sheet_name} ---\n"
                text += sheet_df.to_string(index=False) + "\n\n"
            return text

        else:
            raise ValueError(f"Unsupported file format: {ext}")

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreBuilder:
    def __init__(self, chunk_size=800, chunk_overlap=80, persist_dir="api_faiss_index"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._vectorstore = None

    def split_documents(self, documents):
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Created {len(split_docs)} chunks.")
        return split_docs

    def create_or_load_vectorstore(self, documents):
        persist_path = os.path.join(os.getcwd(), self.persist_dir)

        if os.path.exists(persist_path):
            print("Vectorstore found. Loading from disk...")
            vectorstore = FAISS.load_local(
                folder_path=persist_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("No existing vectorstore. Creating a new one...")
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            vectorstore.save_local(folder_path=persist_path)
            print("Vectorstore saved to disk.")

        self._vectorstore = vectorstore
        return vectorstore

    def get_vectorstore(self):
        if self._vectorstore is not None:
            print("Vectorstore already loaded in memory.")
            return self._vectorstore
        
        print("Loading vectorstore into memory...")
        persist_path = os.path.join(os.getcwd(), self.persist_dir)
        self._vectorstore = FAISS.load_local(
            folder_path=persist_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self._vectorstore

if __name__ == "__main__":
    file_path = "/Users/macbook/Desktop/fliz_Mcp_Rag/api.txt"
    uploader = FileUploader()
    builder = VectorStoreBuilder()

    with open(file_path, "rb") as f:
        text = uploader.extract_text_from_upload(filename=file_path, file_obj=f)

    document = Document(page_content=text, metadata={"source": file_path})
    split_docs = builder.split_documents([document])
    vectorstore = builder.create_or_load_vectorstore(split_docs)

    print("FAISS vectorstore is ready to use.")