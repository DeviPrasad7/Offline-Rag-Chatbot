# Offline RAG Chatbot (Fully Offline)

Offline RAG Chatbot is a **local Retrieval-Augmented Generation (RAG) chatbot** built using Hugging Face models and LangChain.
It runs entirely offline, allowing you to query documents without any internet connection.

---

## Features

* Conversational chatbot interface using **Streamlit**.
* Upload PDF documents for local question answering.
* Uses **FLAN-T5** for text generation and **all-MiniLM-L6-v2** for embeddings.
* Fully offline — no calls to OpenAI or external APIs.
* Formats lab values automatically (e.g., `ng/mL 0.81 - 3.85 0.68`).
* Chat history with timestamps and a clear button in the sidebar.

---

## Folder Structure

```
C:\DocRAG\RAGChat
│
├─ download_models.py     # Downloads FLAN-T5 and embeddings models
├─ start.py                 # Streamlit chatbot interface
├─ models/                # Folder where downloaded models are saved
│   ├─ flan-t5-base/
│   └─ all-MiniLM-L6-v2/
└─ README.md
```

> Note: Large model files (`.safetensors`) are **ignored in Git** to avoid exceeding GitHub limits.

---

## Setup

1. **Clone or download this repository**
2. **Create a Python virtual environment** (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

3. **Install dependencies**:

```powershell
pip install --upgrade pip
pip install transformers sentence-transformers torch streamlit langchain faiss-cpu pytz langchain-community
```

---

## Download Models

Run the model downloader script:

```powershell
python download_models.py
```
Do this with internet turned on, this will download the required libraries.
This will save the models locally in `./models/`:

* FLAN-T5 → `./models/flan-t5-base`
* Sentence-Transformers → `./models/all-MiniLM-L6-v2`

---

## Run Chatbot

Start the Streamlit app:

```powershell
streamlit run start.py
```
* Will run offline , you can turn of your wifi.
* The web interface will open at `http://localhost:8501`.
* Upload PDFs or type questions directly in the chat input.

---

* Large `.safetensors` files are **ignored in Git** using `.gitignore`.

---

## License

MIT License

---

## Acknowledgements

* [Hugging Face](https://huggingface.co/) for open-source models.
* [LangChain](https://www.langchain.com/) for RAG pipelines.
* [Streamlit](https://streamlit.io/) for the interactive interface.
