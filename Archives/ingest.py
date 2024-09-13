
import re
import os
from typing import Callable, List, Tuple, Dict
from embed import EmbeddingGenerator
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
def extract_metadata_from_dict(news_metadata):
        if not news_metadata: return {}

        return {
            "title": news_metadata.get("title", "").strip(),
            "author": news_metadata.get("publisher", "").strip(),
            "creation_date": get_date()
        }

def parse_news(news_dict):
    if not news_dict: return news_dict

    metadata = extract_metadata_from_dict(news_dict)
    text = source.download_and_parse(news_dict.get("link", None)).text

    return metadata, text


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
def clean_texts(chats:list[str], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for idx, text in enumerate(chats):
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((idx, text))
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

def text_to_docs_texts(text: str, metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "page_number": 1,
                "chunk": i,
                "source": f"p1-{i}",
                **metadata,
            },
        )
        doc_chunks.append(doc)

    return doc_chunks

def ingest_any_file(file_path, collection_name):
    if file_path.endswith(".pdf"):
        ingest_new_file(file_path, collection_name)
    elif file_path.endswith(".txt"):
        ingest_text_file(file_path,collection_name)
    else:
        ingest_an_url(file_path, collection_name)

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


def ingest_news_object(news_object):
    # Step 1: Parse PDF
    metadata, text= parse_news(news_object)
    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text_str(text, cleaning_functions)

    document_chunks = text_to_docs_texts(cleaned_text_pdf, metadata)

    # Optional: Reduce embedding cost by only using the first 23 pages
    # document_chunks = document_chunks[:10]

    # Step 3 + 4: Generate embeddings and store them in DB
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=os.getenv("default_finance_collection_name"),
        persist_directory=os.getenv("default_data_directory"),
    )

    # Save DB locally
    vector_store.persist()

def ingest_an_url(url, collection_name):
    # Step 1: Parse PDF
    text = source.download_and_parse(url).text
    metadata = {
            "title": url,
            "author":"/ingest endpoint",
            "creation_date": get_date()
        }
    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text_str(text, cleaning_functions)

    document_chunks = text_to_docs_texts(cleaned_text_pdf, metadata)

    # Optional: Reduce embedding cost by only using the first 23 pages
    # document_chunks = document_chunks[:10]

    # Step 3 + 4: Generate embeddings and store them in DB
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=os.getenv("default_data_directory"),
    )

    # Save DB locally
    vector_store.persist()
def ingest_text_file(file_path, collection_name):
    # Step 1: Parse PDF
    with open(file_path, "r", errors="ignore", encoding="utf8") as f:
        text = f.read()
    
    metadata = {
            "title": file_path,
            "author":"/ingest endpoint",
            "creation_date": get_date()
        }
    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text_str(text, cleaning_functions)

    document_chunks = text_to_docs_texts(cleaned_text_pdf, metadata)

    print(document_chunks)

    # Step 3 + 4: Generate embeddings and store them in DB
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=os.getenv("default_data_directory"),
    )

    # Save DB locally
    vector_store.persist()
    print("Sucessfully Ingested text to ", collection_name)
    

def ingest_chat_for_a_user(chats:list, user_id:str):
    # Step 1: Parse PDF
    metadata = {
        "user_id": user_id,
        "ingestion_date":  get_date()
    }
    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_texts(chats, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # Optional: Reduce embedding cost by only using the first 23 pages
    # document_chunks = document_chunks[:10]

    # Step 3 + 4: Generate embeddings and store them in DB
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name=os.getenv("default_collection_name") if user_id is None or "" else user_id,
        persist_directory=os.getenv("default_data_directory")
    )

    # Save DB locally
    vector_store.persist()
    print('Ingestion Sucessfull')


if __name__ == "__main__":

    previous_chats =  [
    "Hello my name is Prajwal", 
    "I live in a grand palace situated in Mumbai", 
    "I own the company called Padmraj industries",
    "I have a younger brother, who is still studying"
    "My wife is the managing Director of Padmraj industries",
    "I love playing chess and understanding more about people", 
    "I am keen towards financial understading"
]

    ingest_new_file("0001166691-23-000024.pdf", collection_name="Praj")


