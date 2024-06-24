from langchain_community.llms import Ollama
from langchain_community import embeddings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


st.markdown(
    """
    <style>
    .header {
        display: flex;
        align-items: center;
        padding: 20px;
    }
    .header img {
        max-width: 100px; 
        margin-top: 12px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns((2.5,10))

with col1:
    st.markdown(
        """
        <div class="header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/09/Mastek_logo.png" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.title("Demo Mastek Chatbot")

st.markdown("---")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model="mistral")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory
    )
    return conversation_chain

with st.sidebar:
    st.subheader("your doc")
    pdf_docs = st.file_uploader("choose here", accept_multiple_files=True)
    if st.button("process"):
        with st.spinner("processing"):

            raw_text = get_pdf_text(pdf_docs)

            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)

            vectorstore = get_vectorstore(text_chunks)

            st.session_state.conversation = get_conversation_chain(vectorstore)
            
        
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
prompt = st.chat_input("What is up?")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = response['chat_history']
    latest_answer = response['answer']

    with st.chat_message("assistant"):
        st.markdown(latest_answer)
    st.session_state.messages.append({"role": "assistant", "content": latest_answer})


    # PRINITNG THE RERANK AND SIMILARITY SEARCH

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = Ollama(model="mistral")
    def pretty_print_docs(docs):
        st.write("Relevant documents")

        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write(f"Metadata: {doc.metadata}")
            st.write("\n" + "-" * 100 + "\n")
    retriever_result = retriever.get_relevant_documents(prompt)
    pretty_print_docs(retriever_result)

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )
    def pretty_print_docs(docs):
        st.write("Reranked documents")
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write(f"Metadata: {doc.metadata}")
            st.write("\n" + "-" * 100 + "\n")
    compression_result = compression_retriever.get_relevant_documents(prompt)
    pretty_print_docs(compression_result)



