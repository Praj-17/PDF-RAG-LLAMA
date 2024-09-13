# RAG-Based Chat with PDF Application

## Overview
This application uses the Retrieval Augmented Generation (RAG) model to enable chatting with the content of PDF documents. It extracts text from PDFs, generates embeddings, and stores them in a vector store. The RAG model then retrieves relevant passages from the vector store to generate answers to user questions.

## Architecture
The application consists of the following main components:

1. **PDF Text Extraction**: Uses PyMuPDF and pdfplumber to extract text from PDF files, handling encrypted PDFs and extracting metadata like title, author, and creation date.

2. **Text Cleaning and Ingestion Pipeline**: Cleans the extracted text using functions like merging hyphenated words, fixing newlines, and removing multiple newlines. It then splits the text into chunks and converts them to LangChain Documents with metadata. Finally, it ingests the documents into a Chroma vector store.

3. **Chat Model and Retriever**: Uses the LangChain library to create a RetrievalQA chain with a custom MyVectorStoreRetriever. The retriever uses the Chroma vector store to find relevant passages based on a similarity score threshold and a specified number of top results.

4. **Generative Response Creation**: The RAG model generates answers to user questions by combining information from the retrieved passages. It appends the page numbers of the relevant passages to the generated answer.

## Retrieval Approach
The retrieval process works as follows:

1. The user provides a question or query.
2. The custom MyVectorStoreRetriever uses the Chroma vector store to find the most relevant passages based on the similarity score threshold and the specified number of top results.
3. The retrieved passages are used as context to generate an answer to the user's question using the RAG model.
4. The generated answer is returned along with the page numbers of the relevant passages.

## Usage
To use the application, follow these steps:

1. Install the required libraries using pip: `pip install -r requirements.txt`
2. Set the necessary environment variables, such as the Chroma vector store parameters and the OpenAI API key in the `example.env` file. Rename the file `example.env` to `.env`
3. Ingest PDF files into the vector store using the `ingest_new_file` function.
4. Initialize the `Chat` class and use the `chat` method to ask questions and retrieve answers.

Example usage:

```python
chat = Chat(collection_name="my_collection")
ingest_new_file("path/to/pdf_file.pdf", collection_name="my_collection")
question = "What are some usage of samples in biomedical research?"
answer, relevant_pages = chat.chat(question)
print(answer)
print(relevant_pages)
```

This code initializes the `Chat` class, ingests a PDF file into the vector store, asks a question, and prints the generated answer along with the relevant page numbers.

## Limitations and Future Improvements
- The current implementation uses a fixed similarity score threshold and number of top results for retrieval. Dynamically adjusting these parameters based on the quality of the retrieved passages could improve the accuracy of the generated answers.
- Handling large PDF files efficiently, such as by processing them in batches or using incremental indexing, could improve the scalability of the application.
- Incorporating user feedback to fine-tune the RAG model and improve its performance on specific domains or types of questions could enhance the overall user experience.

## Conclusion
This RAG-based chat application demonstrates how to leverage the power of retrieval-augmented generation to enable interactive conversations with the content of PDF documents. By combining text extraction, embedding generation, vector store indexing, and generative language modeling, the application provides a flexible and extensible platform for knowledge-based question answering.
