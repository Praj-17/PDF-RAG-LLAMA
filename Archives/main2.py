import os
import re
from typing import List, Tuple, Dict, Callable
from datetime import datetime
from dotenv import load_dotenv

# PDF processing libraries
import pdfplumber
import PyPDF4

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

def get_date():
    """Returns the current date as a string."""
    return datetime.today().strftime('%Y-%m-%d')

def extract_metadata_from_pdf(file_path: str) -> dict:
    """Extracts metadata from a PDF file."""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)
        if not reader.isEncrypted:
            metadata = reader.getDocumentInfo()
            return {
                "title": metadata.get("/Title", "").strip(),
                "author": metadata.get("/Author", "").strip(),
                "creation_date": metadata.get("/CreationDate", "").strip(),
            }
        else:
            return {
                "title": "Unknown Title",
                "author": "Unknown Author",
                "creation_date": get_date(),
            }

def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """Extracts text from each page of a PDF file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((page_num + 1, text))
    return pages

def merge_hyphenated_words(text: str) -> str:
    """Merges words that are hyphenated at line breaks."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def fix_newlines(text: str) -> str:
    """Replaces single newlines with spaces."""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def remove_multiple_newlines(text: str) -> str:
    """Removes multiple consecutive newlines."""
    return re.sub(r"\n{2,}", "\n", text)

def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    """Applies cleaning functions to the text of each page."""
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages

def text_to_docs(
    text: List[Tuple[int, str]], metadata: Dict[str, str]
) -> List[Document]:
    """Converts text into LangChain Document objects with metadata."""
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=200,
    )
    for page_num, page_text in text:
        chunks = text_splitter.split_text(page_text)
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
    """Ingests a new PDF file into the Chroma vector store."""
    # Step 1: Parse PDF
    raw_pages = extract_pages_from_pdf(file_path)
    metadata = extract_metadata_from_pdf(file_path)

    # Step 2: Clean text
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)

    # Step 3: Create document chunks
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # Step 4: Generate embeddings and store them in the vector store
    embeddings = OpenAIEmbeddings()
    persist_directory = os.getenv("default_data_directory")  # Default directory for storing the vector database
    vector_store = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    vector_store.persist()

class Chat:
    def __init__(self, collection_name="default_collection"):
        """Initializes the chat model with a specified collection."""
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.collection_name = collection_name
        self.persist_directory = os.getenv("default_data_directory")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )
        retriever = self.vector_store.as_retriever(
            search_type=os.getenv("search_type"),  # You can adjust the search type if needed
            search_kwargs={"k": int(os.getenv("top_k_to_search"))},    # Number of top documents to retrieve
        )
        # Initialize the language model and retrieval chain
        llm = OpenAI(model_name="4omini")  # Use '4omini' as the chat model
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=os.getenv("chain_type"),  # Adjust as needed; options include 'map_reduce', 'refine', etc.
            retriever=retriever,
            return_source_documents=True,
        )

    def chat(self, question, chat_history=[]):
        """Generates an answer to the question using the retrieval chain."""
        response = self.chain({"query": question, "history": chat_history})
        answer = response["result"]
        source_documents = response["source_documents"]
        # Extract text samples from source documents
        text_samples = [doc.page_content for doc in source_documents]
        return answer, text_samples

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Specify the collection name
    collection_name = "Praj"

    # Ingest a new PDF file into the vector store
    pdf_file_path = "Data\IARC Sci Pub 163_Chapter 3.pdf"
    if os.path.exists(pdf_file_path):
        ingest_new_file(pdf_file_path, collection_name)
    else:
        print(f"PDF file not found at path: {pdf_file_path}")

    # Initialize the chat model
    chat = Chat(collection_name=collection_name)

    # Chat history can be maintained across multiple interactions
    chat_history = []

    # Define your question
    question = "What are some usages of samples in biomedical research and laboratory practices?"

    # Get the answer and source document texts
    answer, text_samples = chat.chat(question=question, chat_history=chat_history)

    # Update chat history if needed
    chat_history.append({"question": question, "answer": answer})

    # Print the answer and source documents
    print("Answer:", answer)
    print("\nSource Documents:")
    for idx, text in enumerate(text_samples):
        print(f"\n--- Document {idx + 1} ---\n{text}")
