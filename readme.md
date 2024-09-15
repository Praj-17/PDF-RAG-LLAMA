

## Note:
**Please note that I am using my friends laptop for a while since my laptop has been put to repair and hence, the commits that are made are from her github username. However I assure you this is my original work. I am open for any questions related to it**

# Notebook 
Find the collab notebook [here](https://colab.research.google.com/drive/1mCsG-fxhHmQtrZQjit0DfG9c_XoTT-At?usp=sharing).
# Documentation for PDF Processing and Chat Model Architecture

This documentation outlines the architecture, approach to retrieval, and the generation of responses in the chat model. It also provides setup instructions for running the application.

## Model Architecture

The application consists of several components that work together to process PDF documents, extract relevant information, and generate responses to user queries. The main components are:

1. **PDF Processing**:
   - **Metadata Extraction**: Uses `PyPDF4` to extract metadata such as title, author, and creation date from PDF files.
   - **Text Extraction**: Utilizes `pdfplumber` to extract text from each page of the PDF.
   - **Text Cleaning**: Applies various cleaning functions to ensure the extracted text is formatted correctly (e.g., merging hyphenated words, fixing newlines).

2. **Document Chunking**:
   - The extracted and cleaned text is split into smaller chunks using `RecursiveCharacterTextSplitter` from LangChain. This allows for better handling of large texts during retrieval and response generation.

3. **Vector Store**:
   - The cleaned document chunks are stored in a Chroma vector store. This allows for efficient retrieval of relevant documents based on user queries.

4. **Retrieval and Response Generation**:
   - The chat model uses `ChatOpenAI` from LangChain to generate responses. It retrieves relevant documents from the vector store based on the user’s query and generates an answer using the retrieved content.

## Approach to Retrieval

The retrieval process involves the following steps:

1. **Query Processing**: When a user submits a question, the query is processed to identify relevant documents from the vector store.
2. **Document Retrieval**: The system retrieves the top `k` documents that are most relevant to the query using the specified search type (e.g., cosine similarity).
3. **Response Generation**: The retrieved documents are fed into the language model, which generates a coherent response based on the content of those documents.

## Generative Responses

The generative responses are created by leveraging the capabilities of the `ChatOpenAI` model. The model synthesizes information from the retrieved documents and formulates an answer that is contextually relevant to the user's question. The response is designed to be informative and engaging, providing insights based on the content of the source documents.

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.
- An OpenAI API key. You can obtain one from [OpenAI](https://platform.openai.com/account/api-keys).

## Setup Instructions

To set up the application, follow these steps:

1. **Clone the Repository**: Clone the repository containing the code to your local machine from the following link

   ```
   https://github.com/Praj-17/PDF-RAG-LLAMA
   ```

2. **Install Dependencies**: Ensure you have Python installed, and then install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Environment File**:
   - There is an example environment file named `example.env` in the repository. 
   - Rename this file to `.env` and edit it to include your OpenAI API key. The file should look like this:

     ```
     OPENAI_API_KEY=your_openai_api_key
     default_data_directory=path_to_your_data_directory
     search_type=default_search_type  # e.g., 'similarity'
     top_k_to_search=5  # Number of top documents to retrieve
     chain_type=default_chain_type  # e.g., 'map_reduce'
     ```

4. **Pull The Docker Image**:
   ```
   docker pull prajwal1717/interactive-pdf-qna-chatbot:latest
   ```
5. **Run the Docker image with the `.env` file **:
   ```
   docker run -d -p 8501:8501 --env-file .env prajwal1717/interactive-pdf-qna-chatbot:latest
   ``` 

5. **Acess the app**

Open your browser and navigate to http://localhost:8501.

6. **Interact with the Chat Model**: After running the application, you can ask questions based on the ingested PDF content, and the model will provide responses along with relevant source documents.

## Conclusion

This documentation provides a comprehensive overview of the model architecture, retrieval approach, and setup instructions necessary to run the PDF processing and chat model application. By following the instructions, you can successfully set up and interact with the model to gain insights from your PDF documents.

## References

Find the collab notebook [here](https://colab.research.google.com/drive/1mCsG-fxhHmQtrZQjit0DfG9c_XoTT-At?usp=sharing).