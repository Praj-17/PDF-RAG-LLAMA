from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List
from embed import EmbeddingGenerator
from langchain_community.llms import DeepInfra
from langchain.chains import LLMChain
import math
from dotenv import load_dotenv

import os
load_dotenv()
os.environ["DEEPINFRA_API_TOKEN"]  = os.getenv("DEEPINFRA_API_KEY")



class MyVectorStoreRetriever(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )

        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs

prompt = """You are {name}. Immerse yourself in this character provided in backticks , incorporating their distinct personality, preferences, and communication style in your interactions. Respond authentically as the character, reflecting their unique traits, mannerisms, and interests in conversations.

```
Character Profile:
profile bio: {profile.get("bio")}
Interests: {existing_persona.get("interests")}, {profile.get("interests")}
knowledge Areas: {profile.get("knowledgeAreas")}
Personality Traits: {existing_persona.get("personality_traits")}, {profile.get("personalityTraits")}
Hobbies: {existing_persona.get("hobbies")}
Skills: {existing_persona.get("skills")}
Values: {existing_persona.get("values")}
Emotions: {existing_persona.get("emotions")}
Age Group: {existing_persona.get("age_group")}
Favorites: {profile.get("favourites")}
Gender: {existing_persona.get("gender")}
```
Please be mindful of the following:
1. Avoid assuming aspects of {name}'s personality unless explicitly provided.
2. Should personal inquiries arise without sufficient information, reply with a polite disclaimer, suggesting additional data be shared.
3. Responses should not be very long, it should look like a healthy conversation

[User Input]: {question}
[Your Response as {name}]:
"""


class Chat:
    def __init__(self, collection_name=""):
        '''
        return ConversationalRetrievalChain.from_llm(
            model,
            #retriever=vector_store.as_retriever(),
            retriever = MyVectorStoreRetriever(
                vectorstore=vector_store,
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.2, "k": 3},
            ),
            return_source_documents=True,
            # verbose=True,
        )
    '''
        self.embedding = EmbeddingGenerator()
        self.collection_name=collection_name
        self.vector_store = Chroma(
            collection_name=os.getenv("default_collection_name") if self.collection_name == "" or None else self.collection_name,
            embedding_function=self.embedding,
            persist_directory=os.getenv("default_data_directory"),
        )
        self.chain = RetrievalQA.from_chain_type(
            DeepInfra(model_id=os.getenv("CHAT_MODEL_NAME")),
            chain_type=os.getenv("chain_type"),
    
            retriever = MyVectorStoreRetriever(
                vectorstore=self.vector_store,
                search_type=os.getenv("search_type"),
                search_kwargs={"score_threshold": float(os.getenv("score_threshold")), "k": int(os.getenv("top_k_to_search"))},
            ),
            return_source_documents=True,
        )
    
    def chat(self, question, chat_history = [], collection_name=None):
        answer = None
        response = self.chain({"query": question, "history":chat_history})
        answer = response["result"]
        source = response["source_documents"]

        pgs = []
        for document in source:
            pgs.append(document.metadata['page_number'])
            #print(f"List after inserting:", pgs)
            
        for i in range(0, len(pgs)):
            for j in range(i+1, len(pgs)):
                #if(l[i] == l[j]):
                if(math.isclose(pgs[i], pgs[j], abs_tol = 2)):
                        pgs.insert(0, pgs[i])
        pgs = list(set(pgs))
        return answer, pgs
    
        



    
if __name__ == "__main__":
    load_dotenv()
    chat = Chat(collection_name="Praj")
    chat_history = [prompt]

    while True:
        print("Within While Loop")
        question = input("Question: ")
        answer, pgs = chat.chat(question = question, collection_name="Praj", chat_history=[prompt] )
        chat_history.append(answer)
        print(answer, pgs)


       