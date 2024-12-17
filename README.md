
# Chat with PDF Using RAG Pipeline

This project is a PDF scraping application built with Streamlit that extracts text, table data, and images (using OCR) from uploaded PDF files. It utilizes **Large Language Models (LLMs)** for answering user queries based on the extracted content, powered by vector embeddings.

---

## Features

- **PDF Upload**: Upload one or more PDF files through the Streamlit sidebar.
- **Text Extraction**: Extracts plain text, table data, and image text using OCR.
- **Image OCR**: Processes images like graphs or pie charts inside PDFs and extracts textual content using `pytesseract`.
- **LLM Integration**: Uses the Groq-based LLM (`llama-3.1-70b-versatile`) to answer user questions about the processed data.
- **Vector Storage**: Embeds the extracted text into vector format using `HuggingFaceEmbeddings` and stores it in a FAISS index.
- **Query Processing**: Retrieve and display answers to user queries with a retrieval-augmented generation (RAG) approach.

---

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

