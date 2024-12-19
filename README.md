
# Chat with PDF Using RAG Pipeline
## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to process, analyze, and interact with semi-structured data from multiple PDF files. The system extracts text, images, and tables, embeds the data into a vector database, and enables accurate retrieval and response generation using a selected LLM.

## Features

- *
## Features
1. **Data Ingestion**  
   - Extract text and structured data from uploaded PDF files.  
   - Segment data into logical chunks for granular embedding.  
   - Convert chunks into vector embeddings using a pre-trained model.  
   - Store embeddings in a FAISS vector database for efficient retrieval.  

2. **Query Handling**  
   - Accept user queries in natural language.  
   - Perform similarity searches in the vector database to find relevant chunks.  
   - Use an LLM to generate responses based on retrieved chunks.  

3. **Comparison Queries**  
   - Handle user queries requiring comparisons across PDFs.  
   - Retrieve and aggregate relevant data for comparison.  
   - Provide structured responses (e.g., tables, bullet points).  

4. **Response Generation**  
   - Leverage retrieval-augmented prompts to ensure context-rich, factual responses

## Technologies Used

### Libraries and Frameworks
- `streamlit`: UI for the web app.
- `pdfplumber`: Extracts tables and text from PDFs.
- `PyMuPDF` (`fitz`): Handles PDF content like images.
- `pdfminer.six`: Extracts plain text from PDFs.
- `langchain`: Builds a retrieval-based question-answering chain.
- `pytesseract`: OCR for extracting text from images.
- `HuggingFaceEmbeddings`: Converts text into embeddings.
- `FAISS`: Efficient vector storage for retrieval-based systems.
- `pickle`: Saves and reloads the FAISS index.

### LLM and Models
- **Groq LLM**: `llama-3.1-70b-versatile` model for answering questions.
- **HuggingFace Transformers**: For text embedding using `sentence-transformers/all-MiniLM-L6-v2`.
### Install dependencies
  pip install -r requirements.txt
### Run the Application
streamlit run Task1.py

### Task1 Output
![Task1 Output](https://i.postimg.cc/QMKw7ZPH/Screenshot-262.png)
---
# Chat with Websites using RAG Pipeline

This project leverages web scraping, sentence embedding, and a language model to provide concise answers to user queries about websites.

## Features
- **Web Crawling and Scraping**: Extracts content from university websites.
- **Sentence Embedding**: Processes and stores content embeddings using `SentenceTransformer`.
- **Efficient Search**: Implements FAISS for fast and accurate similarity searches.
- **Query Handling**: Matches user queries with the most relevant content.
- **Response Generation**: Uses GPT-2 to generate answers in a human-readable format.

## Project Structure
1. **DataIngestion**:
   - Crawls and scrapes web pages.
   - Segments content and generates embeddings.
   - Stores embeddings using FAISS for efficient similarity search.

2. **QueryHandler**:
   - Handles user queries.
   - Searches for relevant content chunks using FAISS.

3. **ResponseGenerator**:
   - Generates context-based answers using GPT-2.

### Task2 Output
![Task2 Output](https://i.postimg.cc/Kj1MGQ4S/output2.jpg)

## Setup Instructions

### Prerequisites
- Python 3.9 or later.
- Tesseract OCR installed:
  - On **Linux**: `sudo apt install tesseract-ocr`
  - On **Windows**: [Download and install Tesseract](https://github.com/tesseract-ocr/tesseract)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone (https://github.com/varalu2004/Sithafal_Assessment)

