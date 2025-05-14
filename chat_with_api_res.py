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

app = FastAPI()

# Initialize once
api_handler = FlizApiHandler(api_doc_path="/Users/macbook/Desktop/fliz_Mcp_Rag/api.txt")
groq_client = Groq()

class QueryRequest(BaseModel):
    query: str
    method: str = "GET"

@app.post("/query")
async def query_api(request: QueryRequest):
    # Step 1: Get the API info using LangChain + vector search
    result = api_handler.process_query(request.query, method=request.method)

    # Step 2: Prepare message for Groq LLM
    try:
        stream = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that analyzes API responses for users.\n"
                        "Your task is to:\n"
                        "- Extract only the most relevant and necessary data.\n"
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

        # Step 3: Return both raw API result and Groq summary
        return {
            "query_result": result,
            "summary": description.strip()
        }

    except Exception as e:
        return {
            "error": f"Groq streaming failed: {str(e)}",
            "query_result": result
        }

@app.get("/ping")
async def ping():
    return {"message": "API is up and running!"}


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chat_with_api_res:app", host="0.0.0.0", port=port, reload=True)
