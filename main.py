# import os
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain


# # Load environment variables
# load_dotenv()

# working_dir = os.path.dirname(os.path.abspath(__file__))


# def load_document(file_path):
#     loader = PyPDFLoader(file_path)
#     documents = loader.load()
#     return documents


# def setup_vectorstore(documents):
#     embeddings = HuggingFaceEmbeddings()
#     text_splitter = CharacterTextSplitter(
#         separator="\n",  # Fixed separator
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     doc_chunks = text_splitter.split_documents(documents)
#     vectorstore = FAISS.from_documents(doc_chunks, embeddings)
#     return vectorstore


# def create_chain(vectorstore):
#     llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0
#     )

#     retriever = vectorstore.as_retriever()
#     memory = ConversationBufferMemory(
#         llm=llm,
#         output_key="answer",
#         memory_key="chat_history",
#         return_messages=True
#     )
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         chain_type="map_reduce",
#         memory=memory,
#         verbose=True
#     )
#     return chain


# # Streamlit configuration
# st.set_page_config(
#     page_title="Chat with Doc",
#     page_icon="📄",
#     layout="centered"
# )

# st.title("📚 PDF Guru: Ask Away! 🤖")

# # Initialize the chat history in streamlit session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

# if uploaded_file:
#     file_path = f"{working_dir}/{uploaded_file.name}"
    
#     # Save the uploaded file
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Initialize the vector store only once
#     if "vectorstore" not in st.session_state:
#         documents = load_document(file_path)
#         st.session_state.vectorstore = setup_vectorstore(documents)

#     # Initialize the conversation chain only once
#     if "conversation_chain" not in st.session_state:
#         st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User input field
# user_input = st.chat_input("Ask Llama...")

# if user_input:
#     # Append user message to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})

#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Get assistant's response and append to chat history
#     with st.chat_message("assistant"):
#         try:
#             response = st.session_state.conversation_chain({"question": user_input})
#             assistant_response = response["answer"]
#         except Exception as e:
#             assistant_response = f"An error occurred while processing your question: {e}"

#         st.markdown(assistant_response)
#         st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})







import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain

# Streamlit configuration
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

st.title("📚 PDF Guru: Ask Away! 🤖")

# 🔄 Reset Chat Button
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.session_state.conversation_chain = None
    st.success("Chat history and session have been reset!")

# Initialize the chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload file
uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process document and embeddings
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        with st.spinner("Processing your PDF and creating embeddings..."):
            documents = load_document(file_path)
            st.session_state.vectorstore = setup_vectorstore(documents)
            st.success("✅ File uploaded and embeddings stored!")

    # Setup chain
    if "conversation_chain" not in st.session_state or st.session_state.conversation_chain is None:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        st.info("🚀 Ready! You can now start asking questions about your PDF.")

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask Llama...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response["answer"]
        except Exception as e:
            assistant_response = f"An error occurred while processing your question: {e}"

        st.markdown(assistant_response)
