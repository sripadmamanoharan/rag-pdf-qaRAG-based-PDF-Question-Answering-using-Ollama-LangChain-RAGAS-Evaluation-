import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 📊 RAGAS Evaluation Imports
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset

# Step 1: Read PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Step 2: Split into chunks
def split_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Step 3: Create FAISS vectorstore
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

# Step 4: Create RAG chain using Ollama
def create_rag_chain(vectorstore):
    llm = OllamaLLM(model="llama3")  # Updated import
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Step 5: Ask and evaluate
def ask_and_evaluate(chain, query):
    print("🔍 Getting model response...")
    response = chain.invoke(query)              # Returns dict
    result = response["result"]                 # Extract answer
    print(f"\n🧠 Answer: {result}")              # ✅ Only print result

    print("📥 Retrieving context documents...")
    try:
        docs = chain.retriever.get_relevant_documents(query)
        context = [doc.page_content for doc in docs]
        print(f"📄 Retrieved {len(context)} chunks.")
    except Exception as e:
        print("❌ Failed to get context:", e)
        return

    print("🧪 Preparing data for RAGAS...")
    try:
        data = Dataset.from_dict({
            "question": [query],
            "answer": [result],        # ✅ Make sure this is a plain string
            "contexts": [context],
            "ground_truths": [[]]
        })
    except Exception as e:
        print("❌ Failed to prepare dataset:", e)
        return

    print("📊 Running RAGAS evaluation...")
    try:
        scores = evaluate(data, metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ])
        print("\n📈 Evaluation Results:")
        print(scores)
    except Exception as e:
        print("❌ Evaluation failed:", e)


# Main
if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ")
    query = input("Ask a question: ")

    print("🔍 Reading PDF...")
    text = extract_text(pdf_path)

    print("📖 Splitting into chunks...")
    chunks = split_chunks(text)

    print("📦 Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    print("🤖 Running LLM RAG chain...")
    rag_chain = create_rag_chain(vectorstore)

    ask_and_evaluate(rag_chain, query)
