import os, tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Google Gemini Q&A Chatbot",
    page_icon="🤖"
)

# Create a subheader for the Streamlit app
st.subheader('🤖 Generative Q&A with LangChain & Gemini')
            
# Sidebar section to input the Google API key
with st.sidebar:
    if 'GOOGLE_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        api_key = st.secrets['GOOGLE_API_KEY']
    else:
        api_key = st.text_input('Enter Google API Key:', type='password')
        if not (api_key.startswith('AI')):
            st.warning('Please enter your API Key!', icon='⚠️')
        else:
            st.success('Success!', icon='✅')
    os.environ['GOOGLE_API_KEY'] = api_key

    # Display external links in the sidebar
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/wms31/langchain-pdf-gemini)"
    "[Check out the blog post!](https://letsaiml.com/create-your-own-ai-chat-using-gemini-for-free/)"

# Input section to upload a source document and provide a query
source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")
query = st.text_input("Enter your query")

if st.button("Submit"):
    # Validate inputs
    if not source_doc or not query:
        st.warning(f"Please upload the document and provide the missing fields.")
    else:
        try:
            # Save uploaded file temporarily to disk, load and split the file, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pdf_documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=0)
            pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
            pdfs = splitter.split_text(pdf_context)
            os.remove(tmp_file.name)
            
            # Specify the directory path for persistence of data
            persist_directory = "./db/gemini"
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Create vector database from text chunks and persist it
            vectordb = Chroma.from_texts(pdfs, embeddings, persist_directory=persist_directory)
            vectordb.persist()

            # Create a chatbot model using Gemini Pro and perform question-answering using RetrievalQA
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, convert_system_message_to_human=True)
            qa = RetrievalQA.from_chain_type(llm=model, retriever=vectordb.as_retriever())
            response = qa.run(query)
            
            st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")