# 📚 Simple PDF RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, ChromaDB, Sentence Transformers, and the Groq API.

## Features

* Upload a PDF document
* Extract text from the PDF
* Split the text into chunks
* Generate embeddings using Sentence Transformers
* Store embeddings in ChromaDB
* Retrieve the most relevant chunks based on the user's question
* Generate answers using Groq Llama 3.1

## Tech Stack

* Python
* Streamlit
* Groq API
* ChromaDB
* Sentence Transformers
* PyPDF2

## Project Structure

```
project/
│── rag2.py
│── requirements.txt
└── README.md
```

## Installation

1. Clone the repository.

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Groq API key.

Replace the API key in the code or use an environment variable.

4. Run the application.

```bash
streamlit run rag2.py
```

## How It Works

1. Upload a PDF.
2. The PDF text is extracted.
3. The text is divided into smaller chunks.
4. Embeddings are created for each chunk.
5. Chunks are stored in ChromaDB.
6. When you ask a question, the most relevant chunks are retrieved.
7. Groq Llama 3.1 uses the retrieved context to generate an answer.

## Example

**Question**

```
What is Machine Learning?
```

**Answer**

```
Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data without being explicitly programmed.
```

## Future Improvements

* Support multiple PDFs
* Chat history
* Persistent ChromaDB storage
* Better text chunking
* Source citations
* Conversation memory

## License

This project is for learning and educational purposes.
