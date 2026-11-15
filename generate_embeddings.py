import os 
import dspy
import numpy as np
import asyncio
from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


async def gemini_embed(texts: list[str]):
    response = await asyncio.to_thread(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=texts,
    )
    embeddings = [item.values for item in response.embeddings]
    return np.array(embeddings, dtype=np.float32)

# if __name__ == "__main__":
#     # Example usage
#     texts = ["What is the capital of France?", "Paris is in France."]
#     embeddings = asyncio.run(gemini_embed_async(texts))
#     print(embeddings.shape) 
