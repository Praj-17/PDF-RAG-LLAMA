

import fitz
import os
class PDFtoText():
    def __init__(self) -> None:
        pass

    def open_pdf(self, pdf):
        if os.path.exists(str(pdf)) or isinstance(pdf,bytes):
                self.pdf = fitz.open(pdf)
                self.page_count = self.pdf.page_count
                return self.pdf
                # self.pdf.close()
        else:
            raise ValueError(f"PDF path is incorrect", pdf)
    def extract_all_text(self, pdf):
         # Open the PDF file
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)

        all_text = ''

        # Iterate through all pages
        for page_number in range(self.page_count) :
            # Get the page
            page = self.pdf[page_number]

            # Extract text from the page
            text = page.get_text()
            all_text += text

            # Print or process the extracted text as needed
            # print(f"Page {page_number + 1}:\n{text}\n")
        return all_text

    def extract_all_text_page_wise(self, pdf):
         # Open the PDF file
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)

        all_text = []

        # Iterate through all pages
        for page_number in range(self.page_count) :
            # Get the page
            page = self.pdf[page_number]

            # Extract text from the page
            text = page.get_text()
            all_text.append(text)
        return all_text
    def extract_text_from_single_page(self,pdf, page_number):
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)
        if page_number -1> self.page_count:
             raise ValueError("Invlaid pagenumber")
        else:
             return self.pdf[page_number-1].get_text()
    def extract_text_from_interval(self,pdf,page_number, interval =1):
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)
        text = ""
        if page_number > self.page_count:
            raise ValueError("Invlaid pagenumber")
        else:
            # Calculate the start and end pages
            start_page = max(0, page_number - interval)
            end_page = min(self.page_count - 1, page_number + interval)

            for page_number in range(start_page, end_page + 1):
                text += self.extract_text_from_single_page(pdf=pdf, page_number=page_number)
        return text

"""### Embedding Generation using DeepInfra Models"""

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

"""### Text Cleaning and Ingestion Pipeline"""

import re
import os
from typing import Callable, List, Tuple, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import pdfplumber
import os
import PyPDF4
load_dotenv()
os.getenv("DEEPINFRA_API_KEY")

embeddings = EmbeddingGenerator()
from datetime import datetime

def get_date():
# Get today's date
    today_date = datetime.today()

# Format today's date as a string
    return today_date.strftime('%Y-%m-%d')


def extract_metadata_from_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)  # Change this line
        if not reader.isEncrypted:
            metadata = reader.getDocumentInfo()
        else:
            metadata = {
            "title": "Title",
            "author": "Author",
            "creation_date": get_date(),
        }

        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }
def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata
def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata
def extract_metadata_from_dict(news_metadata):
        if not news_metadata: return {}

        return {
            "title": news_metadata.get("title", "").strip(),
            "author": news_metadata.get("publisher", "").strip(),
            "creation_date": get_date()
        }


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)

def clean_text_str(text:str, cleaning_functions: List[Callable[[str], str]]):
    for cleaning_function in cleaning_functions:
            result = cleaning_function(text)

    return result
def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages
def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
    for page_num, page in text:
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks

def ingest_new_file(file_path, collection_name):
    # Step 1: Parse PDF
    raw_pages, metadata = parse_pdf(file_path)

    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]

    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # Optional: Reduce embedding cost by only using the first 23 pages
    # document_chunks = document_chunks[:10]

    # Step 3 + 4: Generate embeddings and store them in DB
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=os.getenv("default_collection_name") if collection_name is None or "" else collection_name,
        persist_directory=os.getenv("default_data_directory")
    )

"""### The Chat model and the Retriever"""

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List
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
    ingest_new_file("/content/IARC Sci Pub 163_Chapter 3.pdf", collection_name="Praj")
    chat_history = []
    question = "what are some usage of samples, Biomedical research and laboratory practices"
    answer, pgs = chat.chat(question = question, collection_name="Praj", chat_history=[] )
    chat_history.append(answer)
    print(answer, pgs)

