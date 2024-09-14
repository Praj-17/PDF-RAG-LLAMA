# app.py
import streamlit as st
import os
from main import ingest_new_file, get_answer  # Replace 'your_script_name' with the name of your script
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
st.title("Interactive PDF Q&A Chatbot")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_ingested" not in st.session_state:
    st.session_state.file_ingested = False

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None and not st.session_state.file_ingested:
    # Save the uploaded file to a temporary directory
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File {uploaded_file.name} uploaded successfully.")

    # Ingest the file into the vector store
    collection_name = "user_uploaded_docs"
    with st.spinner("Processing the PDF..."):
        ingest_new_file(file_path, collection_name)
    st.info("File processed and ingested into the vector store.")

    st.session_state.file_ingested = True

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the uploaded PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the file has been ingested
    if not st.session_state.get("file_ingested", False):
        st.error("Please upload and process a PDF file first.")
    else:
        # Process the user's question
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_data = get_answer(prompt, collection_name="user_uploaded_docs")

                answer = answer_data['answer']
                st.markdown(answer)

                # Display the relevant document segments
                st.subheader("Relevant Document Segments:")
                for doc in answer_data['documents']:
                    with st.expander(f"Segment {doc['index'] + 1}"):
                        st.write(doc['text'])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
