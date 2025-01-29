import os
from PyPDF2 import PdfReader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Chunk text for embeddings (using sentence segmentation)
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If the current chunk is not empty and adding the sentence exceeds the chunk_size, add the chunk
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    # Add the last chunk if any
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Step 3: Generate embeddings using T5-small (for answering questions)
def generate_embeddings(chunks, batch_size=4):
    model_name = "google/flan-t5-large"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            # Extract embeddings from the model's encoder
            encoder_outputs = model.get_encoder()(**inputs)
            embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            all_embeddings.append(embeddings)  # Store the embeddings

        # Clear CUDA cache to avoid memory overflow
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)  # Return a single array with all embeddings

# Step 4: Create FAISS index and save chunks
def create_faiss_index(chunks, faiss_index_path):
    embeddings = generate_embeddings(chunks)

    # Ensure the embeddings dimensionality matches FAISS expectations (512 for t5-small)
    dimension = embeddings.shape[1]  # This should be 512 for t5-small
    assert dimension == 1024, f"Expected dimensionality 512, but got {dimension}."

    # Create a FAISS index for the embeddings
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean distance)
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, f"{faiss_index_path}_faiss_index.index")

    # Save the original chunks (for reference)
    with open(f"{faiss_index_path}_generated_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and generated chunks saved successfully!")

if __name__ == "__main__":
    pdf_path = "data/labour_law.pdf"
    faiss_index_path = "faiss_store/faiss_index"

    # Extract, chunk, and index
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)  # Prepare chunks using sentence segmentation
    create_faiss_index(chunks, faiss_index_path)
    print("FAISS index created and saved successfully!")
