# # use_fliz_api.py
# from simpleRag import FlizApiHandler

# api_handler = FlizApiHandler(api_doc_path="/Users/macbook/Desktop/MCP_S/mcpWithApiData/api.txt")
# result = api_handler.process_query("Show me rental partner details of arrow costruction", method="GET")

# import json
# print(json.dumps(result, indent=4))


# from groq import Groq
# client = Groq()

# # Your JSON response string (can also load from a file or API)

# stream = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant who analyzes delivery service data. and generate detailed description"
#         },
#         {
#             "role": "user",
#             "content": f"""Here is a list of delivery companies in JSON format:
# {result}"""
#         }
#     ],
#     model="llama-3.3-70b-versatile",
#     temperature=0.5,
#     max_completion_tokens=1024,
#     top_p=1,
#     stop=None,
#     stream=True,
# )

# for chunk in stream:
#     print(chunk.choices[0].delta.content or "", end="")


from fastapi import FastAPI
from pydantic import BaseModel
from simpleRag import FlizApiHandler
from groq import Groq
import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Initialize components
api_handler = FlizApiHandler(api_doc_path="/Users/macbook/Desktop/fliz_Mcp_Rag/api.txt")
groq_client = Groq()
logging.info("Initialized FlizApiHandler and Groq client.")

class QueryRequest(BaseModel):
    query: str
    method: str = "GET"

@app.post("/query")
async def query_api(request: QueryRequest):
    logging.info(f"Received query: {request.query}")
    logging.info(f"HTTP Method: {request.method}")

    # Step 1: Process query with LangChain + vector search
    start_time = time.time()
    result = api_handler.process_query(request.query, method=request.method)
    logging.info(f"API processing completed in {time.time() - start_time:.2f}s")
    logging.info("Raw API result:")
    logging.info(json.dumps(result, indent=2))

    # Step 2: Call Groq LLM to summarize
    try:
        logging.info("Starting Groq LLM streaming response...")
        stream = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Your task is to:\n"
                        "- Extract only the most relevant and necessary data.\n"
                        "- Do not include image URLs, banner URLs, or any visual media links.\n"
                        "- If available, include the `url` field from the data (e.g., 'data.url') to help the user access the service.\n"
                        "- Remove all unnecessary technical fields such as IDs, timestamps, metadata, etc.\n"
                        "- Format the output using clean, labeled sections or bullet points.\n"
                        "- Keep it human-readable and structured (not JSON).\n"
                        "- Do not add explanations or hallucinated content.\n"
                        "Present the response as if you're summarizing it for a user in a clear, factual way."

                    )
                },
                {
                    "role": "user",
                    "content": f"""Here is the raw API response:\n{json.dumps(result, indent=2)}"""
                }
            ],
            model="Llama3-8b-8192",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
        )

        description = ""
        for chunk in stream:
            content_piece = chunk.choices[0].delta.content or ""
            description += content_piece

        logging.info("Groq summary generated successfully.")
        logging.info(f"Summary:\n{description.strip()}")

        return {
            "query_result": result,
            "summary": description.strip()
        }

    except Exception as e:
        logging.error("Groq streaming failed", exc_info=True)
        return {
            "error": f"Groq streaming failed: {str(e)}",
            "query_result": result
        }

@app.get("/ping")
async def ping():
    logging.info("Received ping request.")
    return {"message": "API is up and running!"}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("chat_with_api_res:app", host="0.0.0.0", port=port)

