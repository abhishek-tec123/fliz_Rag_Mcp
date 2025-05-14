# endpoit reterever and api response with modularity and importable --------------------------------------------------------------

import os
import glob
import json
import time
import logging
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Environment Variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
auth_token = os.getenv('AUTH_TOKEN')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
FAISS_INDEX_PATH = "api_faiss_index"
BASE_URL = "http://192.168.29.55:3000/"

# Prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Retrieve the most relevant endpoint(s) from the provided context based on the user's query. "
    "Make sure to differentiate between 'rental' and 'delivery' when processing the request. "
    "Respond only with a valid JSON object. Do not include any explanation or extra text. "
    "The format must be:\n\n"
    "{{\n"
    "  \"endpoint_url\": \"<URL>\",\n"
    "  \"payload\": {{\n"
    "    \"field1\": \"<value>\",\n"
    "    \"field2\": \"<value>\",\n"
    "    \"field3\": \"<value>\"\n"
    "  }}\n"
    "}}\n\n"
    "Context: {context}\n\nQuery: {input}"
)

class EndpointRetriever:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vectors = None

    def vector_embedding(self):
        embeddings = HuggingFaceEmbeddings()
        if self.vectors:
            return {"status": "FAISS index already loaded."}

        if os.path.exists(FAISS_INDEX_PATH) and glob.glob(f"{FAISS_INDEX_PATH}/*"):
            self.vectors = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return {"status": "Loaded existing FAISS index."}

        with open(self.file_path, 'r') as file:
            text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vectors = FAISS.from_documents(documents, embeddings)
        self.vectors.save_local(FAISS_INDEX_PATH)

        return {"status": "Vector embeddings created and saved."}

    def retrieval(self, query):
        if not self.vectors:
            return None

        retriever = self.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': query})
        return response.get("answer", None)

class FlizApiHandler:
    def __init__(self, api_doc_path):
        self.retriever = EndpointRetriever(file_path=api_doc_path)
        self.vector_status = self.retriever.vector_embedding()

    def _hit_api(self, full_url, payload, method='GET'):
        headers = {
            "Authorization": f"{auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            if method.upper() == 'POST':
                response = requests.post(full_url, headers=headers, json=payload)
            else:
                response = requests.get(full_url, headers=headers, params=payload)

            response.raise_for_status()
            return response.status_code, response.json()
        except requests.exceptions.RequestException as e:
            return None, {"error": str(e)}

    def process_query(self, query, method='GET'):
        start_time = time.time()
        model_raw_response = self.retriever.retrieval(query)

        try:
            model_response = json.loads(model_raw_response)
            endpoint_url = model_response.get("endpoint_url")
            payload = model_response.get("payload")
        except Exception as e:
            return {
                "error": f"Model response parsing failed: {e}",
                "raw_model_response": model_raw_response,
                "vector_status": self.vector_status
            }

        if endpoint_url and payload:
            full_url = BASE_URL + endpoint_url
            status_code, api_response = self._hit_api(full_url, payload, method)
        else:
            return {
                "error": "Missing endpoint_url or payload in model response.",
                "raw_model_response": model_raw_response,
                "vector_status": self.vector_status
            }

        response_time = time.time() - start_time

        return {
            "query": query,
            "vector_status": self.vector_status,
            "endpoint_url": endpoint_url,
            "payload": payload,
            "full_url": full_url,
            "model_raw_response": model_raw_response,
            "api_status_code": status_code,
            "api_response": api_response,
            "response_time": response_time
        }
