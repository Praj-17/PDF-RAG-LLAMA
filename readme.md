# RAG-Based Chat Application with PDF Processing

## Overview
This application leverages Retrieval Augmented Generation (RAG) to enable interactive querying of PDF documents. It extracts text and metadata from PDFs, cleans and processes the text, and stores it in a vector database. The application utilizes a language model to generate responses based on the retrieved content, allowing users to ask questions and receive informative answers.

## Architecture
The application consists of the following key components:

1. **PDF Processing**: 
   - **Text Extraction**: Utilizes `pdfplumber` and `PyPDF4` to extract text from PDF files, handling page-wise extraction and metadata retrieval.
   - **Text Cleaning**: Implements functions to clean the extracted text by merging hyphenated words, fixing newlines, and removing excessive whitespace.

2. **Document Ingestion**:
   - Converts cleaned text into `Document` objects from the LangChain library, which include metadata such as page numbers and source information.
   - Stores the documents in a Chroma vector store, allowing for efficient retrieval.

3. **Retrieval Mechanism**:
   - Implements a custom retriever that uses similarity search to find relevant documents based on user queries.
   - Configurable parameters allow for adjusting the number of top documents to retrieve and the similarity score threshold.

4. **Generative Response Creation**:
   - Utilizes the OpenAI language model to generate answers based on the retrieved documents.
   - Combines the retrieved context with the user’s query to create coherent and relevant responses.

## Approach to Retrieval
The retrieval process follows these steps:

1. **User Query**: The user submits a question related to the content of the PDF.
2. **Similarity Search**: The application uses the custom retriever to perform a similarity search in the Chroma vector store, retrieving the top documents that are most relevant to the query.
3. **Response Generation**: The retrieved documents are passed to the language model, which generates a response based on the context provided by the documents.
4. **Output**: The application returns the generated answer along with the page numbers of the documents used to formulate the response.

## Usage
To use the application, follow these steps:

1. **Install Required Libraries**: Ensure you have the necessary libraries installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**:
   - Create a file named `example.env` in the root directory of your project.
   - Add your DeepInfra API key to the `example.env` file as follows:
     ```
     DEEPINFRA_API_KEY=your_deepinfra_api_key
     ```
   - Rename `example.env` to `.env` to load the environment variables correctly.

3. **Ingest PDF Files**: Use the `ingest_new_file` function to process and store a PDF document.
   
4. **Initialize the Chat Model**: Create an instance of the `Chat` class to prepare for querying.

5. **Ask Questions**: Use the `chat` method to submit your queries and receive answers.

### Example Usage
```python
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Specify the collection name
    collection_name = "Praj"

    # Ingest a new PDF file into the vector store
    ingest_new_file("path/to/your/pdf_file.pdf", collection_name)

    # Initialize the chat model
    chat = Chat(collection_name=collection_name)

    # Define your question
    question = "What are some usages of samples in biomedical research and laboratory practices?"

    # Get the answer and source document texts
    answer, text_samples = chat.chat(question)

    # Print the answer and source documents
    print("Answer:", answer)
    print("\nSource Documents:")
    for idx, text in enumerate(text_samples):
        print(f"\n--- Document {idx + 1} ---\n{text}")
```

## Conclusion
This RAG-based chat application effectively combines PDF processing, text cleaning, and retrieval-augmented generation to provide an interactive platform for querying document content. By leveraging advanced language models and efficient retrieval mechanisms, users can obtain precise and contextually relevant answers to their questions.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/30051062/dc00c957-061c-4e7c-a7d5-332053404fa1/paste.txt