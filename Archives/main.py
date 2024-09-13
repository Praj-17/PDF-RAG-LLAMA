import os
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv
load_dotenv()


class EmbeddingFunction:
    def __init__(self) -> None:
        self.openai = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
    def __call__(self, input):
        embedding = list(self.openai.embeddings.create(
            model="BAAI/bge-large-en-v1.5",
            input=input,
            encoding_format="float"
        ))

        return dict(embedding[0][1][0])['embedding']
class ChromaStore:
    def __init__(self) -> None:
        # Configure Chroma with a persistent SQLite database
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embed_function = EmbeddingFunction()

    def embed_documents(self, texts: list, user_id: str):
        embeddings = []
        collection = self.client.get_or_create_collection(user_id)

        for idx, text in enumerate(texts):
            embedding = self.embed_function.__call__(text)
            embeddings.append(embedding)
        
        collection.add(ids= [f"{i}" for i in range(len(texts))], embeddings= embeddings, metadatas=[{"text": text} for text in texts])
        # Persist the data
        # self.client.persist()

        return embeddings
    def query_documents(self, query, user_id):
        try:
            collection = self.client.get_collection(user_id)
            embeddings = self.embed_function.__call__(query)
            docs = collection.query(query_embeddings = embeddings, n_results=5)
            metadatas = docs['metadatas'][0]
            output = [metadata['text'] for metadata in metadatas]
        except Exception as e:
            output = []
        
        return output






if __name__ == "__main__":
    user_id = "233"

    # List of text samples
    previous_chats =  [
    "Hello my name is Prajwal", 
    "I live in a grand palace situated in Mumbai", 
    "I own the company called Padmraj industries",
    "I have a younger brother, who is still studying"
    "My wife is the managing Director of Padmraj industries",
    "I love playing chess and understanding more about people", 
    "I am keen towards financial understading"
]

    # Generating and storing embeddings with persistence
    encoder = ChromaStore()
    embeddings = encoder.embed_documents(previous_chats, user_id)
    print("embedding done")

    docs = encoder.query_documents(query="Where is my palace", user_id=user_id)
    print(docs)
