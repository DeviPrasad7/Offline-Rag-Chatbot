import os
import re
from datetime import datetime

import pytz
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

os.environ["HUGGINGFACE_HUB_TOKEN"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

LOCAL_EMBEDDING_MODEL = "./models/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "./models/flan-t5-base"

st.set_page_config(page_title="ProRAG Chatbot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stDecoration"] { display: none; }
header, footer, #MainMenu { visibility: hidden; height: 0; }
div.block-container { padding-top: 0.5rem; }
body { background-color: #1e1e1e; color: #ffffff; }
.chat-row { display: flex; flex-direction: column; margin-bottom: 15px; animation: fadeIn 0.3s ease-in; }
.chat-bubble {
  padding: 14px 20px; border-radius: 25px; max-width: 70%;
  word-wrap: break-word; font-size: 15px; line-height: 1.5; color: #ffffff;
}
.user-bubble { background: linear-gradient(135deg, #005f99 0%, #007acc 100%); margin-left: auto; margin-right: 10px; }
.bot-bubble  { background: linear-gradient(135deg, #111 0%, #1a1a1a 100%); margin-right: auto; margin-left: 10px; }
.timestamp { font-size: 12px; color: #888; margin: 2px 10px; }
.bot-timestamp { text-align: left; margin-left: 10px; }
.user-timestamp { text-align: right; margin-right: 10px; }
.header { display: flex; align-items: center; gap: 15px; margin-bottom: 15px; padding: 10px; }
.header img { width: 50px; height: 50px; }
.header h1 { color: #00bfff; margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.query-section { background: #1a1a1a; padding: 12px !important; border-radius: 8px; text-align: center; margin-bottom: 10px !important; }
.query-section p { color: #ffffff; font-size: 18px; font-weight: bold; margin: 0 !important; padding: 0 !important; line-height: 1.1; }
.status-box { padding: 10px; border-radius: 8px; background-color: #2a2a2a; color: white; font-size: 14px; }
@keyframes fadeIn { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);} }
[data-testid="stSpinner"] > div {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
  padding: 0 !important;
}
.stApp[data-teststate=running] .stChatInput textarea,
.stApp[data-test-script-state=running] .stChatInput textarea { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <img src="https://img.icons8.com/ios-filled/50/00bfff/robot.png" alt="ProRAG Icon"/>
  <h1>ProRAG Chatbot</h1>
</div>
<div class="query-section">
  <p>Ask your documents anything</p>
</div>
""", unsafe_allow_html=True)


def clean_lab_text(text: str) -> str:
    pattern = re.compile(r"(ng/mL|mg/dL)\s+(\d+\.\d+)\s*-\s*(\d+\.\d+)\s*(\d+\.\d+)")
    def replacer(m):
        unit = m.group(1)
        ref_low = m.group(2)
        ref_high = m.group(3)
        result = m.group(4)
        return f"Result: {result} {unit}, Reference Range: {ref_low} – {ref_high} {unit}"
    return pattern.sub(replacer, text)


text_data = """
ProRAG Chatbot provides local RAG capabilities.
It can answer questions from uploaded PDFs, DOCX, and TXT files using offline Hugging Face models.
"""


with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose file(s) (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )

    documents = []
    if uploaded_files:
        st.session_state["file_processing"] = True
        with st.spinner("Processing file(s)..."):
            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    ext = file.name.split(".")[-1].lower()
                    if ext == "pdf":
                        loader = PyPDFLoader(temp_path)
                    elif ext == "docx":
                        loader = UnstructuredWordDocumentLoader(temp_path)
                    elif ext == "txt":
                        loader = TextLoader(temp_path)
                    else:
                        st.warning(f"Unsupported file type: {file.name}")
                        continue
                    docs = loader.load()
                    for doc in docs:
                        if doc.page_content:
                            doc.page_content = clean_lab_text(doc.page_content)
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
        st.session_state["file_processing"] = False
    else:
        documents = [text_data]

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)

if documents and hasattr(documents[0], "page_content"):
    chunks = text_splitter.split_documents(documents)
    raw_texts = [c.page_content for c in chunks]
else:
    raw_texts = text_splitter.split_text(str(documents))

embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

vector_store = FAISS.from_texts(raw_texts, embeddings)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL, local_files_only=True)

from langchain_huggingface import HuggingFacePipeline

class TruncatingHuggingFacePipeline(HuggingFacePipeline):
    def _call(self, prompt: str, stop=None):
        inputs = self.pipeline.tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.pipeline.model.generate(
                **inputs, max_length=256
            )
        decoded = self.pipeline.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return decoded[0]

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
)

llm = TruncatingHuggingFacePipeline(pipeline=hf_pipeline)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    verbose=False
)

simple_responses = {
    "thank you": "You're welcome! 😊",
    "thanks": "Happy to help! 😄",
    "hello": "Hi there! How can assistance be provided today? 👋",
    "hi": "Hello! What’s on the mind? 😄",
    "bye": "Goodbye! Have a great day! 👋",
    "goodbye": "Goodbye! Feel free to return anytime! 😊"
}

input_disabled = st.session_state.get("file_processing", False) or st.session_state.get("bot_processing", False)
user_input = st.chat_input("💬 Type your message...", disabled=input_disabled)

def safe_question(q: str) -> str:
    return q[:2000]

if user_input:
    st.session_state["bot_processing"] = True
    ist = pytz.timezone("Asia/Kolkata")
    current_time = datetime.now(ist).strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", user_input, current_time))

    with st.spinner("ProRAG is typing..."):
        user_input_lower = user_input.lower().strip()
        bot_message = simple_responses.get(user_input_lower)
        if not bot_message:
            response = qa_chain.invoke({"question": safe_question(user_input)})
            bot_message = response["answer"]
        st.session_state.chat_history.append(("bot", bot_message, current_time))
    st.session_state["bot_processing"] = False

with st.container():
    for role, msg, timestamp in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
                <div class='chat-row' style='align-items: flex-end;'>
                    <div class='chat-bubble user-bubble'>{msg}</div>
                    <div class='timestamp user-timestamp'>{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-row' style='align-items: flex-start;'>
                    <div class='chat-bubble bot-bubble'>{msg}</div>
                    <div class='timestamp bot-timestamp'>{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)