from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

class EmbeddingGenerator:
    def __init__(self) -> None:
        pass
        # Create an OpenAI client with your deepinfra token and endpoint
        self.openai = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
    def embed_query(self, text:str):
        embedding = self.openai.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input=text,
        encoding_format="float"
        )
        return embedding.data[0].embedding
    def embed_documents(self, texts:list):
        emb = []
        for i in range(len(texts)):
            embedding = self.embed_query(i)
            emb.append(embedding)
        return emb

if __name__ == "__main__":
    gen = EmbeddingGenerator()
    input = ["Prajwal", "loves", "sakshi"]
    embedding =  gen.embed_documents(input)
    print(len(embedding))
    print(len(embedding[0]))
    print(embedding[0])




