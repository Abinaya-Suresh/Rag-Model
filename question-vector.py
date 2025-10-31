import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="")


def ask_question(question):
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("Error: Please run pdf-vector file first")
        return None

    try:
        # Load index + chunks
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data["chunks"]
        metadata = data["metadata"]
        total_pages = data["total_pages"]

        # Encode question
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_vector = embed_model.encode([question])

        # Search FAISS
        scores, indices = index.search(query_vector.astype("float32"), k=3)
        print(f"Found {len(indices[0])} relevant chunks:")

        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page_num = metadata[idx]["estimated_page"]
            context_parts.append(f"[Page {page_num}]: {chunk_text}")

        context = "\n\n".join(context_parts)

        # Generate answer with Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"You are a helpful assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        return response.text

    except Exception as e:
        print("Error:", str(e))
        return None


def main():
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("Missing database files. Please run the PDF vectorization script first.")
        return

    with open("chunks.pkl", "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    total_pages = data["total_pages"]

    while True:
        question = input("Enter your question (or type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question.lower() == "info":
            print("\nDatabase Info:")
            print(f"Total pages: {total_pages}")
            print(f"Total chunks: {len(chunks)}")
            print(f"Vector dimension: {len(chunks)}")
            print(f"Average chunks per page: {len(chunks)//total_pages}")
            print(f"Sample chunk: {chunks[0][:100]}...\n")
            continue

        if not question:
            print("Please enter a question")
            continue

        print("\nSearching and generating answer...")
        answer = ask_question(question)

        if answer:
            print(f"\nAnswer: {answer}\n")
        else:
            print("Sorry, I couldn't generate an answer.\n")


if __name__ == "__main__":
    main()
