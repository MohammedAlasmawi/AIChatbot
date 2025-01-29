# AI Chatbot
# Bahrain Labor Law Chatbot

This is a **Bahrain Labor Law Chatbot**, built using **Django (Backend)** and **Flask (AI Service)**. It leverages **FAISS for similarity search**, **Hugging Face transformers for NLP**, and **PyPDF2 for document processing**. The chatbot provides answers based on **Bahrain's labor law document**.

---

## Features

✅ Extracts text from **PDF files**  
✅ Splits text into **manageable chunks**  
✅ Generates **embeddings** using `FLAN-T5`  
✅ Uses **FAISS** for similarity search  
✅ Provides **accurate answers** using prompt engineering  
✅ Simple **Flask API** for interaction  
✅ **Django Backend** for user management and API integration  

---

## Project Structure


| File / Directory | Description |
|-----------------|-------------|
| `backend/` | Django project for managing user requests. |
| `flask_api/` | Flask service for AI-based query processing. |
| `faiss_store/` | Stores FAISS indexes for fast retrieval. |
| `data/` | Folder containing labor law PDF documents. |
| `frontend/` | Optional frontend UI for chatbot interaction. |

---

## Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/labor-law-chatbot.git
cd labor-law-chatbot


cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

cd ../frontend
pip install -r requirements.txt
python app.py
