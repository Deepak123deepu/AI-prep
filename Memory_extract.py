import dspy
from typing import Dict, List
import asyncio 
from dotenv import load_dotenv 
load_dotenv()
from pydantic import BaseModel
from generate_embeddings import gemini_embed

class Memory(BaseModel):
    information: str
    categories: str

class MemoryExtract(dspy.Signature):
    """Extract memory-related information from the conversation history. 
    create memory entries that you should remember when speaking with user later,
    each memory is a unique automic factoid,
    also make sure to generate categorie for the memeory to help fecting the memory later,"""

    transcript:  List[Dict] = dspy.InputField()
    # information: List[str] = dspy.OutputField(description="have memories in just 4 words")
    # categories: List[str] = dspy.OutputField(description="have memories categorie")
    memoery: List[Memory] = dspy.OutputField(description="have memories in just 4 words and one category for each memory")
    no_info: bool = dspy.OutputField(description="True if no memory information is found, False otherwise")


memory_extractor = dspy.Predict(MemoryExtract)

async def extract_memory_from_messages(transcript: List[Dict]) -> str:
    with dspy.context(lm=dspy.LM("groq/openai/gpt-oss-120b")):
        out = await memory_extractor.acall(transcript=transcript)
    return out.memoery
    # print(f"Memory Extracted: {memory}")

async def embed_memory(memory: List[Memory]):
    
    memory_texts = [m.information for m in memory]
    embeddings = await gemini_embed(memory_texts)
    return embeddings

async def extract_and_embed_memory(messages:List[Dict]):
    memories = await extract_memory_from_messages(messages)
    embeddings = await embed_memory(memories)
    return memories, embeddings


if __name__ == "__main__":
    # Example transcript
    transcript = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"},
        {"role": "user", "content": "I need help with my project. by the way i love python"},
        {"role": "assistant", "content": "Sure, what kind of project is it? python is a great language for many types of projects."},
        {"role": "user", "content": "It's a web development project. I want to build a website using Python and Django."},
        {"role": "assistant", "content": "That sounds great! Django is a powerful framework for web development. Do you have any specific questions or areas you need help with?"},
        {"role": "user", "content": "Yes, I need help with setting up the database and creating models."},
        {"role": "assistant", "content": "I can guide you through that. First, you'll need to configure your database settings in the Django settings file. Then, you can create models by defining classes in your app's models.py file."},
        {"role": "user", "content": "hey wait, i changed my mind, i want to learn AI and ML, can you help me with that?"},
        {"role": "assistant", "content": "Of course! AI and ML are fascinating fields. I can provide you with resources, tutorials, and guidance to get started. What specific area of AI or ML are you interested in?"},
        {"role": "user", "content": "I want to learn about neural networks and deep learning."},
        {"role": "assistant", "content": "Great choice! Neural networks and deep learning are essential components of modern AI. I recommend starting with online courses, tutorials, and hands-on projects to build your understanding. You can also explore libraries like TensorFlow and PyTorch for practical implementation."}
    ]

    memories, embeddings = asyncio.run(extract_and_embed_memory(transcript))
    print(memories)
    print(embeddings.shape)
