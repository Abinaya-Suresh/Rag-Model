import faiss  # for vector storing , it creates file for vector storage
import PyPDF2
import numpy as np
import pickle  # way to save object to files
from sentence_transformers import SentenceTransformer  # for embeddings


def pdf_to_vector(pdf_path):
    print(f"Reading PDF : {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)  # fixed PDFReader -> PdfReader
        total_pages = len(pdf_reader.pages)

        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):  # split into pages
            page_text = page.extract_text()
            page_texts.append({
                'text': page_text,
                'page_number': page_num + 1
            })
        text = ''.join([p['text'] for p in page_texts])  # combine all pages

        print(f"Total pages: {total_pages}")
        print(f"Total text length: {len(text):,} characters")
        print(f"Average characters per page: {len(text)//total_pages:,}")

        chunks = []
        chunk_metadata = []

        for i in range(0, len(text), 400):
            chunk_text = text[i:i+500]
            chunks.append(chunk_text)

            estimated_page = min((i // (len(text)//total_pages)) + 1, total_pages)
            chunk_metadata.append({
                'start_pos': i,
                'estimated_page': estimated_page
            })

        print(f"Created {len(chunks)} chunks")

        print("Getting embeddings from SentenceTransformer (all-MiniLM-L6-v2)")
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embed_model.encode(chunks)

        print("Creating FAISS index")
        embeddings = np.array(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype('float32'))

        print("Saving to files")
        faiss.write_index(index, "vectors.index")
        with open("chunks.pkl", "wb") as f:
            pickle.dump({
                'chunks': chunks,
                'metadata': chunk_metadata,
                'total_pages': total_pages
            }, f)

        print("Vector database created successfully , and files are saved")

        return embeddings, chunks


if __name__ == "__main__":
    pdf_file = r"C:\Users\abina\Desktop\Rag model\casestudy-4.pdf"  
    embeddings, chunks = pdf_to_vector(pdf_file)

    print("\n Setup complete")
