from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from preprocess import extract_text_from_pdf, split_text_into_chunks, generate_embeddings, create_faiss_index

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the FAISS index and chunks
def load_faiss_index(faiss_index_path):
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    # Load the chunks associated with the index
    with open(f"{faiss_index_path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Perform the query on the FAISS index with prompt engineering
def query_faiss_index(query, index, chunks, k=10):
    # Generate embedding for the query
    model_name = "google/flan-t5-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        # Generate embeddings from the encoder output
        query_embedding = model.get_encoder()(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    # Debugging: Print the shape of query embedding and FAISS index dimensionality
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"FAISS index dimensionality: {index.d}")

    # Perform a search on the FAISS index
    if query_embedding.shape[1] != index.d:
        raise ValueError(f"Dimensionality mismatch: Query embedding has shape {query_embedding.shape}, but FAISS index expects dimensionality {index.d}")

    D, I = index.search(query_embedding, k)  # D is the distances, I is the indices of closest chunks

    # Use the retrieved chunks to form a prompt for the model
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
                    You are a legal assistant trained to provide relevant answers to bahrain labor law-related questions based on the content from the "labor_law.pdf" document. Your responses should focus solely on answering the user’s query based on bahrain, extracting the most pertinent information from the provided context. If the document does not contain information directly related to the user's query, inform them politely. Mention the source.

                    Context: 
                    {context}

                    Question: 
                    {query}

                    Answer:
                    (Please extract the most relevant and specific information from the context provided, and formulate a direct answer to the user's query. Provide concise legal advice if the information is available based in bahrain labor law. Mention the source)
                    """
    # Use the model to generate an answer based on the enhanced prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'], 
            max_length=512,  # Increase max_length for longer answers
            do_sample=True,  # Enable sampling for more diverse answers
            temperature=0.7  # Adjust temperature for creativity
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Log the answer sent to the frontend
    print(f"Generated Answer: {answer}")

    return answer

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Load FAISS index and chunks
        print("Loading FAISS index...")  # Debugging line
        faiss_index_path = "faiss_store/faiss_index"
        index, chunks = load_faiss_index(faiss_index_path)
        print("FAISS index loaded.")  # Debugging line
        
        # Get the query from the POST request
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "No query provided!"}), 400

        print(f"Query received: {query}")  # Debugging line
        
        # Get relevant information from the FAISS index using prompt engineering
        answer = query_faiss_index(query, index, chunks)
        print(f"Answer generated: {answer}")  # Debugging line

        # Return the response as a JSON
        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"Error: {e}")  # Debugging line
        return jsonify({"error": "Something went wrong!"}), 500
    
if __name__ == "__main__":
    app.run(port=5001)
