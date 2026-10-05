"""Chat with a PDF using free local models (Ollama).

This is RAG (Retrieval-Augmented Generation), in five steps:
  1. Load      PDF -> text, page by page
  2. Chunk     split text into small overlapping pieces
  3. Embed     turn each chunk into a vector (numbers that capture its meaning)
  4. Retrieve  embed the question, find the chunks whose vectors are closest
  5. Generate  hand those chunks to the LLM and ask it to answer ONLY from them

Run:  python rag.py path/to/file.pdf
"""
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
from pypdf import PdfReader

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
DB = chromadb.PersistentClient(path=str(Path(__file__).parent / "db"))


def post(path, payload):
    req = urllib.request.Request(OLLAMA + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        raise RuntimeError("Can't reach Ollama. Start it with `ollama serve`.") from e


# --- 1 + 2. Load and chunk ---------------------------------------------------
def chunk_text(text, size, overlap):
    """Fixed-size character windows. Overlap stops a sentence that sits on a
    boundary from being cut in half and lost to both chunks."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def load_chunks(pdf_bytes, size, overlap):
    # ponytail: chunks never cross a page, so page citations stay exact;
    # a sentence split across two pages will be split across two chunks.
    chunks = []
    for page_no, page in enumerate(PdfReader(io.BytesIO(pdf_bytes)).pages, start=1):
        text = " ".join((page.extract_text() or "").split())  # collapse whitespace
        chunks += [{"page": page_no, "text": c} for c in chunk_text(text, size, overlap) if c.strip()]
    return chunks


# --- 3. Embed and store ------------------------------------------------------
def embed(texts):
    with post("/api/embed", {"model": EMBED_MODEL, "input": texts}) as r:
        return json.load(r)["embeddings"]


def get_collection(pdf_bytes, size=500, overlap=150):
    """Embed the PDF once and keep the vectors on disk in Chroma. The name comes
    from the file's contents, so a renamed copy is reused and an edited file
    with the same name is re-embedded."""
    name = f"pdf-{hashlib.md5(pdf_bytes).hexdigest()}-{size}-{overlap}"
    col = DB.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        chunks = load_chunks(pdf_bytes, size, overlap)
        if not chunks:
            raise ValueError("No text found. Is this a scanned PDF? It would need OCR first.")
        # nomic-embed-text was trained with these prefixes; using them improves matches
        col.add(ids=[str(i) for i in range(len(chunks))],
                embeddings=embed(["search_document: " + c["text"] for c in chunks]),
                documents=[c["text"] for c in chunks],
                metadatas=[{"page": c["page"]} for c in chunks])
    return col


# --- 4. Retrieve -------------------------------------------------------------
def retrieve(question, col, k=8):  # k and chunk size tuned with eval.py
    res = col.query(query_embeddings=embed(["search_query: " + question]), n_results=min(k, col.count()))
    # Results are lists of lists (one list per question; we sent one, hence [0]).
    # Chroma returns cosine *distance*; similarity = 1 - distance (1 = same meaning).
    return [{"page": m["page"], "text": doc, "score": 1 - d}
            for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


# --- 5. Generate -------------------------------------------------------------
SYSTEM = ("Answer the question using ONLY the context below. Cite pages like [page 3]. "
          "If the answer is not in the context, say \"I can't find that in the document.\"")


def answer(question, hits):
    """Yield the reply piece by piece as the model writes it (streaming)."""
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    with post("/api/chat", {"model": CHAT_MODEL, "messages": messages, "stream": True}) as r:
        for line in r:  # Ollama streams one JSON object per line
            yield json.loads(line)["message"]["content"]


def format_sources(hits):
    return "sources: " + ", ".join(f"p{h['page']} ({h['score']:.2f})" for h in hits)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python rag.py path/to/file.pdf")
    print("Loading (embedding only happens the first time per PDF)...")
    col = get_collection(Path(sys.argv[1]).read_bytes())
    print(f"Ready, {col.count()} chunks. Ask a question (Ctrl+C to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        hits = retrieve(question, col)
        for piece in answer(question, hits):
            print(piece, end="", flush=True)
        print("\n" + format_sources(hits))
