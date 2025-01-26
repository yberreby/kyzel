import logging
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.generate.llm import LLM
from src.types import Conversation
from .cli import parse_args, ServerConfig

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
    filename='server.log'
)
log = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    conversation: Conversation
    max_new_tokens: int = 512

class Server:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI()
        self.llm = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.on_event("startup")
        async def load_model():
            log.info(f"Loading model {self.config.model_name}")
            self.llm = LLM(
                model_name=self.config.model_name,
                chat_template=self.config.chat_template,
                max_seq_length=16000
            )

        @self.app.post("/generate")
        async def generate_handler(request: GenerateRequest):
            response = self.llm.generate(
                request.conversation,
                request.max_new_tokens
            )
            return {"response": response}

    def run(self):
        log.info(f"Starting server on {self.config.host}:{self.config.port}")
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port
        )


if __name__ == "__main__":
    config = parse_args()
    server = Server(config)
    server.run()
