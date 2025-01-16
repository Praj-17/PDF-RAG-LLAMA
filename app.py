import streamlit as st
import os
import time
import re
import json
from src import ingest_new_file, get_answer
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
temp_dir = os.getenv("TEMP_FILE_DIRECTORY")

# Ensure critical environment variables are set
if not openai_api_key:
    st.error("OPENAI_API_KEY is not set. Please provide it as an environment variable.")
    st.stop()

if not temp_dir:
    st.error("TEMP_FILE_DIRECTORY is not set. Please provide it as an environment variable.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("Interactive PDF Q&A Chatbot")

# -----------------------------
# Initialize Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

if "collection_name" not in st.session_state:
    # Default collection name
    st.session_state.collection_name = "default"

# -----------------------------
# Sidebar for PDF Upload
# -----------------------------
with st.sidebar:
    st.header("Upload and Process PDF(s)")

    # Collection name input
    collection_name_input = st.text_input("Enter Collection Name", value=st.session_state.collection_name)

    if 4 <= len(collection_name_input) <= 20:
        # Checkbox to set as default
        if st.checkbox("Set as Default Collection", value=True):
            st.session_state.collection_name = collection_name_input
    else:
        st.error("Collection name must be between 4 and 20 characters.")

    # Multiple PDF uploads
    uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Ensure TEMP_FILE_DIRECTORY exists
        os.makedirs(temp_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            # Only process if file hasn't been ingested before
            if uploaded_file.name not in st.session_state.ingested_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                # Save the uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.info(f"File {uploaded_file.name} uploaded successfully.")

                # Ingest the new file into your vector database
                collection_name = st.session_state.collection_name
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        ingest_new_file(file_path, collection_name)
                        st.success(f"File {uploaded_file.name} processed and ingested.")
                        st.session_state.ingested_files.append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")

                # Optionally remove the file after ingestion
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file {uploaded_file.name}: {e}")

        st.info(f"Total files ingested: {len(st.session_state.ingested_files)}")

    elif st.session_state.ingested_files:
        st.info(f"{len(st.session_state.ingested_files)} file(s) already ingested. Ready to ask questions.")

# -----------------------------
# Define Tabs
# -----------------------------
tab_chat, tab_faqs, tab_samples, tab_contact = st.tabs(["Chat", "FAQs", "Sample Queries", "📞 Contact Me"])

# -----------------------------
# Chat Tab
# -----------------------------
with tab_chat:
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "documents" in message:
                docs = message["documents"]
                if docs:
                    st.subheader("Relevant Document Segments")
                    num_cols = 3
                    cols = st.columns(num_cols)
                    for idx, doc in enumerate(docs):
                        col = cols[idx % num_cols]
                        with col:
                            with st.expander(f"Segment {idx + 1}"):
                                st.write(doc['text'])
                else:
                    st.write("No relevant segments found.")

    # Typing simulation
    def response_generator(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    # Normalize text for greeting detection
    def normalize_text(text):
        return re.sub(r'[^\w\s]', '', text).lower().strip()

    greeting_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hello! What can I do for you?",
        "hey": "Hey there! How can I help?",
        "good morning": "Good morning! What can I assist you with?",
        "good afternoon": "Good afternoon! How can I help you today?",
        "good evening": "Good evening! How may I assist you?"
    }

    # Chat input
    prompt = st.chat_input(placeholder="Hello")
    if prompt:
        # Add user message to conversation
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Greeting check
        normalized_prompt = normalize_text(prompt)
        if normalized_prompt in greeting_responses:
            response = greeting_responses[normalized_prompt]
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        else:
            # Only proceed if files are ingested
            if not st.session_state.ingested_files:
                st.error("Please upload and process at least one PDF file first.")
            else:
                with st.spinner("Thinking..."):
                    collection_name = st.session_state.collection_name
                    try:
                        answer_data = get_answer(prompt, collection_name=collection_name)
                        answer = answer_data['answer']
                    except Exception as e:
                        st.error(f"Error getting answer: {e}")
                        answer = "I'm sorry, I couldn't process your request."

                # Add assistant response to conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "documents": answer_data.get('documents', [])
                })

                # Streaming-like response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for word in response_generator(answer):
                        full_response += word
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                # Display relevant documents, if any
                if answer_data.get('documents'):
                    st.subheader("Relevant Document Segments")
                    num_cols = 3
                    cols = st.columns(num_cols)
                    for idx, doc in enumerate(answer_data['documents']):
                        col = cols[idx % num_cols]
                        with col:
                            with st.expander(f"Segment {idx + 1}"):
                                st.write(doc['text'])
                else:
                    st.write("No relevant segments found.")

# -----------------------------
# FAQs Tab
# -----------------------------
with tab_faqs:
    st.header("FAQs")
    st.markdown("Here are some frequently asked questions:")

    faqs = [
        {
            "question": "How do I upload PDF files?", 
            "answer": "Use the file uploader in the sidebar to upload your PDF files."
        },
        {
            "question": "How do I ask a question about the PDFs?", 
            "answer": "Type your question in the chat input at the bottom of the Chat tab."
        },
        {
            "question": "How do I delete all the ingested data?", 
            "answer": "You will have to delete the 'chroma' folder manually from the code-base."
        },
        {
            "question": "Why are there multiple collections?", 
            "answer": "It’s an optional way to keep different PDFs separated in different vector databases."
        }
    ]

    for faq in faqs:
        with st.expander(f"Q: {faq['question']}"):
            st.write(f"A: {faq['answer']}")

# -----------------------------
# Sample Queries Tab
# -----------------------------
with tab_samples:
    st.header("Sample Queries")

    # Path to example queries JSON file
    json_file_path = "src/constants/example_queries.json"

    # Attempt to load sample queries
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                sample_queries = json.load(f)
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please check the file format.")
            sample_queries = []
    else:
        st.warning(f"JSON file not found at path: {json_file_path}")
        sample_queries = []

    # Provide a link to a sample PDF (make sure this link is correct if you want a direct PDF download)
    pdf_url = "https://github.com/Praj-17/PDF-RAG-LLAMA/blob/main/Data/IARC%20Sci%20Pub%20163_Chapter%203.pdf"
    st.markdown(f"[Download Sample PDF]({pdf_url})")

    if sample_queries:
        st.markdown("Here are some sample queries and their answers created from the above PDF:")
        for query in sample_queries:
            st.subheader(f"Q: {query['question']}")
            st.write(f"A: {query['answer']}")

            if query.get('documents'):
                st.markdown("**Relevant Document Segments:**")
                num_cols = 3
                cols = st.columns(num_cols)
                for idx, doc in enumerate(query['documents']):
                    col = cols[idx % num_cols]
                    with col:
                        with st.expander(f"Segment {doc['index'] + 1}"):
                            st.write(doc['text'])
            else:
                st.write("No relevant segments found.")
    else:
        st.info("No sample queries to display. Please ensure the JSON file path is correct and the file is properly formatted.")

# -----------------------------
# Contact Tab
# -----------------------------
with tab_contact:
    st.header("📞 Contact Information")
    st.write("Feel free to reach out through any of the following platforms 😊:")

    # Email
    st.markdown("### Email")
    st.markdown("[pwaykos1@gmail.com](mailto:pwaykos1@gmail.com)")

    # Phone
    st.markdown("### Phone")
    st.markdown("[7249542810](tel:+17249542810)")

    # LinkedIn / Resume
    st.markdown("### LinkedIn / Resume")
    st.markdown("[![Resume](https://img.icons8.com/doodle/48/000000/resume.png)](https://drive.google.com/file/d/1OiSCu4e_1R7cawKSU80cr63Cd2-4OVq7/view?usp=drivesdk)")

    # GitHub
    st.markdown("### GitHub")
    st.markdown("[![GitHub](https://img.icons8.com/nolan/64/github.png)](https://github.com/praj-17)")
