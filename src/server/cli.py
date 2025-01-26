from dataclasses import dataclass
import argparse

@dataclass
class ServerConfig:
    model_name: str
    chat_template: str
    host: str = "127.0.0.1"
    port: int = 8000

def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Run the generation server")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Phi-4",
        help="Model name/path"
    )

    parser.add_argument(
        "--chat-template",
        type=str,
        default="phi-4",
        help="Chat template name, as recognized by Unsloth"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    args = parser.parse_args()

    return ServerConfig(
        model_name=args.model,
        chat_template=args.chat_template,
        host=args.host,
        port=args.port
    )
