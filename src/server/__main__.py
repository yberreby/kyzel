import logging
from fastapi import FastAPI
from pydantic import BaseModel
from src.generate.llm import LLM
from src.types import Conversation
import uvicorn

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='server.log')
log = logging.getLogger(__name__)

app = FastAPI()

class GenerateRequest(BaseModel):
    conversation: Conversation
    max_new_tokens: int = 512

llm = None

@app.on_event("startup")
async def load_model():
    log.info("Loading model on API startup")
    global llm
    llm = LLM(
        model_name="unsloth/Phi-4",
        max_seq_length=16000
    )

@app.post("/generate")
async def generate_handler(request: GenerateRequest):
    response = llm.generate(request.conversation, request.max_new_tokens)
    return {"response": response}

if __name__ == "__main__":
    # Local bind, use forwarding.
    print("Starting server")
    uvicorn.run(app, host="127.0.0.1", port=8000)
