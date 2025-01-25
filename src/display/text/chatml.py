from src.types import Conversation, ChatMsg

def print_msg(msg: ChatMsg):
    print(f"\n{msg['role'].upper()}:")
    print("-" * 80)
    print(msg["content"])


def print_conversation(conversation: Conversation):
    """
    Simple utility to print a conversation in a readable plaintext format, independently of the tokenizer.
    """

    print("=" * 80)
    for msg in conversation:
        print_msg(msg)
