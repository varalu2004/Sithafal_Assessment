
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

---

## Setup Instructions

### Prerequisites
- Python 3.9 or later.
- Tesseract OCR installed:
  - On **Linux**: `sudo apt install tesseract-ocr`
  - On **Windows**: [Download and install Tesseract](https://github.com/tesseract-ocr/tesseract)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <(https://github.com/varalu2004/Sithafal_Assessment)>

