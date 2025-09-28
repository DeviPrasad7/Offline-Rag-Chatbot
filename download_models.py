import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
os.makedirs("./models", exist_ok=True)
print("Downloading google/flan-t5-base ...")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
t5_model.save_pretrained("./models/flan-t5-base")
t5_tokenizer.save_pretrained("./models/flan-t5-base")
print("Saved FLAN-T5 to ./models/flan-t5-base")
print("Downloading sentence-transformers/all-MiniLM-L6-v2 ...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder.save("./models/all-MiniLM-L6-v2")
print("Saved embeddings model to ./models/all-MiniLM-L6-v2")
print("\n All models are downloaded and saved locally!")