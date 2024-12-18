import os
from dotenv import load_dotenv
load_dotenv()  # Loading all the environment variables
from PIL import Image
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from io import BytesIO

# Configure the API key for the Generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the generative models
model = genai.GenerativeModel("gemini-pro")
image_model = genai.GenerativeModel("gemini-1.5-flash")

# Function to get response from the model with error handling
def get_response(question):
    try:
        response = model.generate_content(question)
        
        # Check if the response contains valid text
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to get image response
def get_image_response(input_text, question):
    if input_text != "":
        response = image_model.generate_content([input_text, question])
    else:
        response = image_model.generate_content(question)
    return response.text

def get_text_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local('faiss_index')

def get_pdf_text(pdf_docs):
    texts = ""
    try:
        for pdf in pdf_docs:
            pdf_stream = BytesIO(pdf.read())  # Wrap the file in BytesIO
            pdf_reader = PdfReader(pdf_stream)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:  # Only add valid text
                    texts += text
    except Exception as e:
        return ""
            
    return texts

# Function to create conversational chain with new model integration
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context". 
    Don't provide a wrong answer. \n\n
    Context: \n{context}\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        # Allow dangerous deserialization flag set to True
        new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {
                "input_documents": docs, "question": user_question
            }, return_only_outputs=True
        )
        return response
    except Exception as e:
        return {"output_text": f"An error occurred while processing your request: {e}"}


# Ensure current_history is initialized in session state
if "current_history" not in st.session_state:
    st.session_state.current_history = []

# Set a default mode, e.g., "Chat Mode"
if "mode" not in st.session_state:
    st.session_state.mode = "Chat Mode"

# Sidebar buttons to select mode
st.sidebar.title("Select Mode")

# Add custom CSS for full-width buttons and center text
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: stretch;
        }
        .sidebar button {
            width: 100%;
            text-align: center;
            font-size: 18px;
            padding: 15px;
        }
        .stButton button {
            width: 100%;
            text-align: center;
            font-size: 18px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar button styles to make them full width and centered
chat_mode_button = st.sidebar.button("Chat Mode")
image_qa_mode_button = st.sidebar.button("Image QA Mode")
pdf_qa_mode_button = st.sidebar.button("PDF QA Mode")

# Update the mode based on button click and reset the history when the mode changes
if chat_mode_button:
    if st.session_state.mode != "Chat Mode":
        st.session_state.mode = "Chat Mode"
        st.session_state.current_history = []  # Clear history when mode changes
elif image_qa_mode_button:
    if st.session_state.mode != "Image QA Mode":
        st.session_state.mode = "Image QA Mode"
        st.session_state.current_history = []  # Clear history when mode changes
elif pdf_qa_mode_button:
    if st.session_state.mode != "PDF QA Mode":
        st.session_state.mode = "PDF QA Mode"
        st.session_state.current_history = []  # Clear history when mode changes

# File upload options in the sidebar
uploaded_file = None
if st.session_state.mode == "Image QA Mode":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"], key="image_upload")
elif st.session_state.mode == "PDF QA Mode":
    uploaded_file = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

# Only show the spinner and process the file if it's uploaded
if uploaded_file is not None and st.session_state.mode == "PDF QA Mode":
    with st.spinner("Processing..."):
        try:
            if uploaded_file is not None:
                raw_text = get_pdf_text(uploaded_file)  # Pass as a list
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_text_vector_store(text_chunks)
                    st.sidebar.success("Done")
                else:
                    pass
            else:
                st.sidebar.error("Please upload a PDF file first!")
        except ValueError as e:
            st.sidebar.error(f"Error: {e}")

# Main interaction area
st.title(f"{st.session_state.mode}")

# Initialize response variable
response = ""

# Display the image immediately after it is uploaded
if uploaded_file is not None and st.session_state.mode == "Image QA Mode":
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

# Input form for new queries
with st.form(key="input_form", clear_on_submit=True):
    col1, col2 = st.columns([8.7, 1.3])
    with col1:
        input_text = st.text_input("", key="input_text", label_visibility="collapsed", placeholder="Ask a question...")
    with col2:
        submit_button = st.form_submit_button("Submit")

    if submit_button and input_text:
        if st.session_state.mode == "Chat Mode":
            response = get_response(input_text)
            # Insert the new question and answer at the beginning of the history
            st.session_state.current_history.insert(0, f"Question: {input_text}")
            st.session_state.current_history.insert(1, f"Answer: {response}")
        elif st.session_state.mode == "Image QA Mode":
            # Get a response based on the image and input text
            response = get_image_response(input_text, image)
            st.session_state.current_history.insert(0, f"Question: {input_text}")
            st.session_state.current_history.insert(1, f"Answer: {response}")
        elif st.session_state.mode == "PDF QA Mode":
            response = user_input(input_text)
            st.session_state.current_history.insert(0, f"Question: {input_text}")
            st.session_state.current_history.insert(1, f"Answer: {response['output_text']}")
        else:
            response = "Please upload a PDF file first!"

# Display the question-answer history (most recent first)
if st.session_state.current_history:
    for i in range(0, len(st.session_state.current_history), 2):
        st.write(st.session_state.current_history[i])  # Display Question
        st.write(st.session_state.current_history[i + 1])  # Display Answer
        st.markdown("---")  # Add a horizontal line between each Q&A
