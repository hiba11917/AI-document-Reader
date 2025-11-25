import os
import numpy as np
import faiss
import json
import requests
import glob
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
GLOBAL_INDEX_PATH = os.path.join(INDEX_DIR, "global.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "global_index_to_chunk_ids.npy")
SQLITE_DB = f"sqlite:///{os.path.join(BASE_DIR, 'metadata.db')}"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MAX_CONTEXT_TOKENS = 3000
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# --- Initialize ---
embedder = SentenceTransformer(EMBEDDING_MODEL)
engine = create_engine(SQLITE_DB, connect_args={"check_same_thread": False})

# Cache for FAISS index and mapping
_cached_index = None
_cached_mapping = None

# ---------------------------
# Build global FAISS index
# ---------------------------
def build_global_index():
    """Merges all per-document FAISS indices into one global index"""
    index_files = glob.glob(os.path.join(INDEX_DIR, "*.index"))
    if not index_files:
        raise ValueError("No document indices found. Run ingest first.")
    
    print(f"Found {len(index_files)} document indices. Building global index...")
    
    # Load first index to get dimension
    first_idx = faiss.read_index(index_files[0])
    dim = first_idx.d
    global_idx = faiss.IndexFlatL2(dim)
    
    mapping = []
    with engine.begin() as conn:
        for idx_file in index_files:
            doc_idx = faiss.read_index(idx_file)
            doc_id = os.path.basename(idx_file).replace(".index", "")
            
            # Get chunk IDs for this document in order
            chunks = conn.execute(text("""
                SELECT id FROM chunks WHERE doc_id = :did ORDER BY chunk_index
            """), {"did": doc_id}).fetchall()
            
            if chunks:
                # Add vectors and maintain mapping
                vectors = doc_idx.reconstruct_n(0, doc_idx.ntotal)
                global_idx.add(vectors.astype('float32'))
                mapping.extend([c[0] for c in chunks])
                print(f"  ✓ Added {len(chunks)} chunks from {doc_id}")
    
    faiss.write_index(global_idx, GLOBAL_INDEX_PATH)
    np.save(MAPPING_PATH, np.array(mapping, dtype=object))
    print(f"✅ Global index built: {global_idx.ntotal} vectors")
    return global_idx, mapping

# ---------------------------
# Load FAISS index and mapping (cached)
# ---------------------------
def get_index_and_mapping():
    """Load with caching to avoid reloading on every query"""
    global _cached_index, _cached_mapping
    
    if _cached_index is None or _cached_mapping is None:
        if not os.path.exists(GLOBAL_INDEX_PATH) or not os.path.exists(MAPPING_PATH):
            raise FileNotFoundError("Global FAISS index or mapping file missing. Run build_global_index() first.")
        _cached_index = faiss.read_index(GLOBAL_INDEX_PATH)
        _cached_mapping = np.load(MAPPING_PATH, allow_pickle=True)
    
    return _cached_index, _cached_mapping

def clear_cache():
    """Clear cached index (useful if index is rebuilt)"""
    global _cached_index, _cached_mapping
    _cached_index = None
    _cached_mapping = None

# ---------------------------
# Retrieve top chunks
# ---------------------------
def retrieve_top_chunks(query, top_k=TOP_K):
    """Retrieve top K most relevant chunks for a query"""
    index, mapping = get_index_and_mapping()
    
    # Embed query
    q_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
    D, I = index.search(q_emb, top_k)
    
    results = []
    with engine.begin() as conn:
        for idx, dist in zip(I[0], D[0]):
            if idx < 0:
                continue
            
            chunk_id = mapping[idx]
            row = conn.execute(text("""
                SELECT c.text, c.page, d.filename
                FROM chunks c JOIN documents d ON c.doc_id = d.id
                WHERE c.id = :cid
            """), {"cid": chunk_id}).fetchone()
            
            if row:
                results.append({
                    "text": row[0],
                    "page": row[1],
                    "filename": row[2],
                    "score": float(dist)
                })
            else:
                print(f"⚠️  Warning: Chunk {chunk_id} not found in database")
    
    return results

# ---------------------------
# Build prompt with token limit
# ---------------------------
def build_prompt(query, retrieved_chunks, max_tokens=MAX_CONTEXT_TOKENS):
    """Build prompt with context, respecting token limits"""
    context_parts = []
    token_count = 0
    
    for r in retrieved_chunks:
        chunk_text = f"Source: {r['filename']} (page {r['page']})\n{r['text']}"
        # Rough estimate: 1 token ≈ 4 characters
        chunk_tokens = len(chunk_text) // 4
        
        if token_count + chunk_tokens > max_tokens:
            print(f"⚠️  Reached token limit. Including {len(context_parts)}/{len(retrieved_chunks)} chunks.")
            break
        
        context_parts.append(chunk_text)
        token_count += chunk_tokens
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided sources.
If the answer isn't in the text, say "I don't know."

Context:
{context}

Question: {query}

Answer (with references to filenames and pages):"""
    
    return prompt.strip()

# ---------------------------
# Ask Ollama (streaming)
# ---------------------------
def ask_ollama(prompt, model=None, timeout=600):
    """Query Ollama with streaming response"""
    if model is None:
        model = LLM_MODEL
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not reach Ollama at {OLLAMA_URL}: {e}")
        return ""
    
    full_text = ""
    try:
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    chunk = data["response"]
                    full_text += chunk
                    print(chunk, end="", flush=True)  # Real-time output
                
                if data.get("done"):
                    break
            except json.JSONDecodeError as e:
                print(f"\n⚠️  Parse error: {e}")
                continue
    except requests.exceptions.ChunkedEncodingError:
        print("\n⚠️  Connection interrupted while streaming")
    
    print()  # Newline after streaming
    return full_text.strip()

# ---------------------------
# RAG Pipeline
# ---------------------------
def rag_query(question):
    """Complete RAG pipeline: retrieve → prompt → generate"""
    print(f"\n🔎 Searching for: {question}")
    
    retrieved = retrieve_top_chunks(question)
    if not retrieved:
        print("❌ No relevant chunks found.")
        return
    
    print(f"✅ Found {len(retrieved)} relevant chunks\n")
    
    prompt = build_prompt(question, retrieved)
    print("--- Prompt Preview ---")
    print(prompt[:500], "...\n")
    
    print("--- Generating Answer ---\n")
    answer = ask_ollama(prompt)
    
    if answer:
        print("\n📚 Sources Used:")
        for i, r in enumerate(retrieved, 1):
            print(f"  {i}. {r['filename']} (page {r['page']}) [similarity: {r['score']:.4f}]")
    else:
        print("❌ No answer generated. Check Ollama connection.")

# ---------------------------
# CLI Interface
# ---------------------------
def main():
    """Interactive CLI for RAG queries"""
    print("=" * 60)
    print("AI Document Reader - RAG Framework")
    print("=" * 60)
    print("Commands:")
    print("  'rebuild' - Rebuild global FAISS index from all documents")
    print("  'quit'    - Exit")
    print("  Or ask a question about your documents")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "rebuild":
            try:
                build_global_index()
                clear_cache()
            except Exception as e:
                print(f"❌ Error building global index: {e}")
            continue
        
        try:
            rag_query(user_input)
        except Exception as e:
            print(f"❌ Error: {e}")

# ---------------------------
# Run from command line
# ---------------------------
if __name__ == "__main__":
    main()