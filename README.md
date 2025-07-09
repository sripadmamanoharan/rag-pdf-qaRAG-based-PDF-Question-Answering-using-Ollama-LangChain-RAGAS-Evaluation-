# RAG-based-PDF-Question-Answering-using-Ollama-LangChain-RAGAS-Evaluation-
RAG-based PDF Question Answering using Ollama + LangChain + RAGAS Evaluation


This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline that answers questions from a PDF using **Ollama (LLaMA3 model)**, **LangChain**, **FAISS vector store**, and evaluates performance using **RAGAS metrics**.

---

## Features

- Extracts text from any PDF file
- Splits text into manageable chunks
- Uses HuggingFace embeddings and FAISS for document search
- Answers questions using the `llama3` model running locally via Ollama
- Evaluates the quality of answers using **RAGAS** metrics like:
  - **Faithfulness**
  - **Answer Relevancy**
  - **Context Precision**

---

## Tech Stack

- Python
- [Ollama](https://ollama.com/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [RAGAS](https://github.com/explodinggradients/ragas)
- HuggingFace Sentence Transformers

---

## Folder Structure

```

rag-pdf-qa/
│
├── rag\_pdf\_qa.py           # Main application
├── README.md               # This file
├── .gitignore              # To exclude venv, **pycache**, PDFs
└── venv/                   # Your Python virtual environment (excluded from Git)

````

---

## Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/sripadmamanoharan/rag-pdf-qaRAG-based-PDF-Question-Answering-using-Ollama-LangChain-RAGAS-Evaluation-.git
cd rag-pdf-qa
````

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# OR manually install:
pip install pymupdf langchain langchain-community langchain-ollama sentence-transformers faiss-cpu ragas datasets
```

### 4. Start Ollama and pull the model

```bash
ollama run llama3
```

Make sure Ollama is running in the background.

---

## How to Run

```bash
python rag_pdf_qa.py
```

Enter the PDF path and your question when prompted.

---

## Sample Output

```bash
Enter PDF file path: sample.pdf
Ask a question: What is the abstract about?

🧠 Answer: Based on the provided context...
📄 Retrieved 4 chunks.
📊 Evaluation Results:
- Faithfulness: 0.91
- Answer Relevancy: 0.88
- Context Precision: 0.85
```

---

## Evaluation

Using `RAGAS`, the model's performance is measured on:

| Metric                | Description                                         |
| --------------------- | --------------------------------------------------- |
| **Faithfulness**      | How truthful the answer is to the retrieved context |
| **Relevancy**         | How relevant the answer is to the query             |
| **Context Precision** | How precise the retrieved content was               |



