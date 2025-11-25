import os
import sys
import uuid
import faiss
import fitz                 # PyMuPDF
import docx
import numpy as np
import pandas as pd
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text
import subprocess
import tempfile

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_files")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
SQLITE_DB = f"sqlite:///{os.path.join(BASE_DIR, 'metadata.db')}"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP = 200

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------
# Database setup
# ---------------------------
engine = create_engine(SQLITE_DB)
meta = MetaData()

docs = Table('documents', meta,
             Column('id', String, primary_key=True),
             Column('filename', String),
             Column('file_hash', String, unique=True),
             Column('n_chunks', Integer),
             Column('origin_path', String))

chunks = Table('chunks', meta,
               Column('id', String, primary_key=True),
               Column('doc_id', String),
               Column('chunk_index', Integer),
               Column('text', Text),
               Column('page', Integer),
               Column('char_start', Integer),
               Column('char_end', Integer))

meta.create_all(engine)

# ---------------------------
# Embedding model (lazy load)
# ---------------------------
_embedder = None

def get_embedder():
    """Lazy load embedding model to avoid loading if not needed"""
    global _embedder
    if _embedder is None:
        print("Loading embedding model... (this may take a minute the first time)")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

# ---------------------------
# Hash & duplicate detection
# ---------------------------
def get_file_hash(filepath):
    """Generate MD5 hash of file for duplicate detection"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def document_exists(file_hash):
    """Check if document already ingested"""
    from sqlalchemy import text
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT id, filename FROM documents WHERE file_hash = :hash
        """), {"hash": file_hash}).fetchone()
        return result

# ---------------------------
# LibreOffice detection
# ---------------------------
def find_soffice():
    """Find LibreOffice executable across common paths"""
    possible_paths = [
        r"C:/Program Files/LibreOffice/program/soffice.exe",
        r"C:/Program Files (x86)/LibreOffice/program/soffice.exe",
        "/usr/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise RuntimeError("LibreOffice not found. Please install it to convert .doc files.")

# ---------------------------
# Text extraction helpers
# ---------------------------
def extract_text_from_pdf(path):
    """Extracts text from each page of a PDF using PyMuPDF"""
    try:
        doc = fitz.open(path)
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append((i + 1, text))
        doc.close()
        return pages
    except Exception as e:
        raise ValueError(f"Error reading PDF {path}: {e}")

def extract_text_from_docx(path):
    """Extracts text from paragraphs of a DOCX file"""
    try:
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])
        return [(-1, text)]
    except Exception as e:
        raise ValueError(f"Error reading DOCX {path}: {e}")

def extract_text_from_doc(path):
    """Converts legacy .doc to .docx using LibreOffice, then extracts text"""
    soffice_path = find_soffice()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run([
                soffice_path, '--headless', '--convert-to', 'docx', path, '--outdir', tmpdir
            ], check=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise ValueError(f"LibreOffice conversion timed out for {path}")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"LibreOffice conversion failed: {e.stderr.decode()}")
        
        # Find the converted file
        converted_file = None
        for f in os.listdir(tmpdir):
            if f.lower().endswith(".docx"):
                converted_file = os.path.join(tmpdir, f)
                break
        
        if not converted_file:
            raise ValueError(f"Failed to convert {path} to DOCX")
        
        return extract_text_from_docx(converted_file)

def extract_text_from_xlsx(path):
    """Extracts text from all sheets in an XLSX file"""
    try:
        xls = pd.ExcelFile(path)
        pages = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            txt = df.astype(str).apply(lambda x: " | ".join(x), axis=1).str.cat(sep="\n")
            pages.append((-1, f"[Sheet: {sheet}]\n{txt}"))
        return pages
    except Exception as e:
        raise ValueError(f"Error reading XLSX {path}: {e}")

def extract_text_from_xls(path):
    """Handles legacy XLS files using xlrd engine"""
    try:
        df = pd.read_excel(path, engine="xlrd")
        txt = df.astype(str).apply(lambda x: " | ".join(x), axis=1).str.cat(sep="\n")
        return [(-1, txt)]
    except Exception as e:
        raise ValueError(f"Error reading XLS file {path}: {e}")

# ---------------------------
# Chunker
# ---------------------------
def chunk_text(page_text, page_num, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks with metadata"""
    text = page_text.strip()
    if not text:
        return []
    
    chunks_list = []
    i = 0
    idx = 0
    
    while i < len(text):
        end = min(len(text), i + chunk_size)
        chunk = text[i:end]
        
        chunks_list.append({
            "page": page_num,
            "text": chunk,
            "char_start": i,
            "char_end": end,
            "chunk_index": idx
        })
        
        idx += 1
        if end == len(text):
            break
        i = end - overlap
    
    return chunks_list

# ---------------------------
# FAISS helpers
# ---------------------------
def create_faiss_index(dim):
    """Create a new FAISS index"""
    return faiss.IndexFlatL2(dim)

def save_faiss(index, path):
    """Save FAISS index to disk"""
    faiss.write_index(index, path)

# ---------------------------
# Main ingest function
# ---------------------------
def ingest_file(filepath):
    """Main ingestion pipeline"""
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found -> {filepath}")
        return None
    
    # Generate file hash for duplicate detection
    print("📋 Computing file hash...")
    file_hash = get_file_hash(filepath)
    
    # Check if already ingested
    existing = document_exists(file_hash)
    if existing:
        print(f"⚠️  Document already ingested!")
        print(f"   ID: {existing[0]}")
        print(f"   Filename: {existing[1]}")
        return existing[0]
    
    fname = os.path.basename(filepath)
    ext = fname.split('.')[-1].lower()
    doc_id = str(uuid.uuid4())
    stored_path = os.path.join(DATA_DIR, f"{doc_id}_{fname}")
    
    # Copy file to storage
    print(f"📂 Storing file...")
    try:
        with open(filepath, "rb") as src, open(stored_path, "wb") as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"❌ Error storing file: {e}")
        return None
    
    # Choose extraction method
    print(f"📄 Extracting text from {ext.upper()}...")
    try:
        if ext == "pdf":
            pages = extract_text_from_pdf(stored_path)
        elif ext == "docx":
            pages = extract_text_from_docx(stored_path)
        elif ext == "doc":
            pages = extract_text_from_doc(stored_path)
        elif ext == "xlsx":
            pages = extract_text_from_xlsx(stored_path)
        elif ext == "xls":
            pages = extract_text_from_xls(stored_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        print(f"❌ Text extraction failed: {e}")
        return None
    
    # Chunk text
    print("✂️  Chunking text...")
    all_chunks, chunk_texts = [], []
    chunk_id_counter = 0
    
    for (page_num, page_text) in pages:
        cdata = chunk_text(page_text, page_num)
        for c in cdata:
            cid = str(uuid.uuid4())
            c["id"] = cid
            c["doc_id"] = doc_id
            c["global_chunk_index"] = chunk_id_counter
            all_chunks.append(c)
            chunk_texts.append(c["text"])
            chunk_id_counter += 1
    
    if not chunk_texts:
        print("❌ No text content extracted!")
        return None
    
    print(f"✅ Created {len(chunk_texts)} chunks")
    
    # Embed chunks
    print(f"🧠 Embedding {len(chunk_texts)} chunks...")
    try:
        embedder = get_embedder()
        embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return None
    
    # Validate embeddings
    if embeddings.shape[0] != len(chunk_texts):
        print(f"❌ Embedding mismatch: got {embeddings.shape[0]} but expected {len(chunk_texts)}")
        return None
    
    # Save FAISS index for this document
    print("💾 Saving FAISS index...")
    try:
        dim = embeddings.shape[1]
        index = create_faiss_index(dim)
        index.add(embeddings.astype('float32'))
        index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
        save_faiss(index, index_path)
    except Exception as e:
        print(f"❌ FAISS index save failed: {e}")
        return None
    
    # Save database records
    print("🗄️  Saving to database...")
    try:
        with engine.begin() as conn:
            conn.execute(docs.insert().values(
                id=doc_id,
                filename=fname,
                file_hash=file_hash,
                n_chunks=len(chunk_texts),
                origin_path=stored_path
            ))
            
            for rec in all_chunks:
                conn.execute(chunks.insert().values(
                    id=rec["id"],
                    doc_id=rec["doc_id"],
                    chunk_index=rec["chunk_index"],
                    text=rec["text"],
                    page=rec["page"],
                    char_start=rec["char_start"],
                    char_end=rec["char_end"]
                ))
    except Exception as e:
        print(f"❌ Database save failed: {e}")
        return None
    
    print(f"\n✅ Ingest complete!")
    print(f"   Document ID: {doc_id}")
    print(f"   File: {fname}")
    print(f"   Chunks: {len(chunk_texts)}")
    print(f"   Location: {stored_path}")
    
    return doc_id

# ---------------------------
# Cleanup function
# ---------------------------
def delete_document(doc_id):
    """Delete a document and its associated index"""
    from sqlalchemy import text
    
    try:
        with engine.begin() as conn:
            # Get document info
            doc = conn.execute(text("""
                SELECT filename, origin_path FROM documents WHERE id = :did
            """), {"did": doc_id}).fetchone()
            
            if not doc:
                print(f"❌ Document not found: {doc_id}")
                return False
            
            # Delete from database
            conn.execute(text("DELETE FROM chunks WHERE doc_id = :did"), {"did": doc_id})
            conn.execute(text("DELETE FROM documents WHERE id = :did"), {"did": doc_id})
        
        # Delete files
        index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
        if os.path.exists(index_path):
            os.remove(index_path)
        
        if os.path.exists(doc[1]):
            os.remove(doc[1])
        
        print(f"✅ Deleted document: {doc[0]}")
        return True
    
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return False

# ---------------------------
# List documents
# ---------------------------
def list_documents():
    """List all ingested documents"""
    from sqlalchemy import text
    
    try:
        with engine.begin() as conn:
            docs_list = conn.execute(text("""
                SELECT id, filename, n_chunks FROM documents ORDER BY filename
            """)).fetchall()
        
        if not docs_list:
            print("No documents ingested yet.")
            return
        
        print("\n" + "=" * 60)
        print("Ingested Documents")
        print("=" * 60)
        for doc_id, filename, n_chunks in docs_list:
            print(f"{filename}")
            print(f"  ID: {doc_id}")
            print(f"  Chunks: {n_chunks}\n")
    
    except Exception as e:
        print(f"❌ Error listing documents: {e}")

# ---------------------------
# Command line interface
# ---------------------------
def main():
    """Interactive CLI for document ingestion"""
    print("=" * 60)
    print("AI Document Reader - Ingestion Pipeline")
    print("=" * 60)
    print("Commands:")
    print("  'ingest <path>'  - Ingest a document")
    print("  'list'           - List all documents")
    print("  'delete <id>'    - Delete a document by ID")
    print("  'quit'           - Exit")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "list":
            list_documents()
            continue
        
        if user_input.lower().startswith("delete "):
            doc_id = user_input[7:].strip()
            delete_document(doc_id)
            continue
        
        if user_input.lower().startswith("ingest "):
            filepath = user_input[7:].strip().strip('"')
            ingest_file(filepath)
            continue
        
        print("Unknown command. Use 'ingest <path>', 'list', 'delete <id>', or 'quit'")

# ---------------------------
# Run from command line
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Interactive mode
        main()
    else:
        # Direct ingest mode
        path = sys.argv[1]
        ingest_file(path)