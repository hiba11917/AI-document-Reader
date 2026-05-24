import json
import os
import re
from collections import defaultdict
from statistics import mean
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_index"
DB_PATH = BASE_DIR / "metadata.db"
SQLITE_DB = f"sqlite:///{DB_PATH}"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", OLLAMA_URL.rsplit("/", 1)[0] + "/tags")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemma3:4b")
TOP_K = int(os.getenv("TOP_K", "1"))
INITIAL_RETRIEVAL_K = int(os.getenv("INITIAL_RETRIEVAL_K", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2200"))
SPREADSHEET_CONTEXT_CHARS = int(os.getenv("SPREADSHEET_CONTEXT_CHARS", "1800"))
SPREADSHEET_LINE_LIMIT = int(os.getenv("SPREADSHEET_LINE_LIMIT", "8"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_CONNECT_TIMEOUT = int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
PRIMARY_MODEL_TIMEOUT = int(os.getenv("PRIMARY_MODEL_TIMEOUT", "60"))
FALLBACK_MODEL_TIMEOUT = int(os.getenv("FALLBACK_MODEL_TIMEOUT", "180"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You answer questions only from the provided context. "
        "If the answer is missing, reply with: I do not know based on the uploaded documents. "
        "If you can answer partially, do not add the fallback sentence after giving a real answer. "
        "Be concise and mention supporting filenames and page numbers naturally."
    ),
)

engine = create_engine(SQLITE_DB, connect_args={"check_same_thread": False})
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _index_path(doc_id):
    return INDEX_DIR / f"{doc_id}.index"


def _mapping_path(doc_id):
    return INDEX_DIR / f"{doc_id}.npy"


def load_document_index(doc_id):
    index_path = _index_path(doc_id)
    mapping_path = _mapping_path(doc_id)
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(f"Index files are missing for document {doc_id}.")
    index = faiss.read_index(str(index_path))
    mapping = np.load(str(mapping_path), allow_pickle=True)
    return index, mapping


def fetch_documents(document_id=None, collection_id=None):
    query = """
        SELECT id, filename, file_type
        FROM documents
        {where_clause}
        ORDER BY created_at DESC
    """
    if document_id:
        where_clause = "WHERE id = :doc_id"
        params = {"doc_id": document_id}
    elif collection_id:
        where_clause = """
            WHERE id IN (
                SELECT doc_id
                FROM collection_documents
                WHERE collection_id = :collection_id
            )
        """
        params = {"collection_id": collection_id}
    else:
        where_clause = ""
        params = {}

    with engine.begin() as conn:
        rows = conn.execute(
            text(query.format(where_clause=where_clause)),
            params,
        ).mappings().all()
    return [dict(row) for row in rows]


def fetch_chunks(chunk_ids):
    if not chunk_ids:
        return {}

    placeholders = ", ".join(f":chunk_id_{index}" for index in range(len(chunk_ids)))
    parameters = {f"chunk_id_{index}": chunk_id for index, chunk_id in enumerate(chunk_ids)}

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT c.id, c.doc_id, c.text, c.page, c.char_start, c.char_end,
                       c.section_label, c.sheet_name, c.cell_range, c.source_ref,
                       d.filename
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE c.id IN ({placeholders})
                """
            ),
            parameters,
        ).mappings().all()

    return {row["id"]: dict(row) for row in rows}


def fetch_spreadsheet_chunks(document_id=None, collection_id=None):
    documents = fetch_documents(document_id=document_id, collection_id=collection_id)
    spreadsheet_ids = [doc["id"] for doc in documents if is_spreadsheet_source(doc.get("filename"))]
    if not spreadsheet_ids:
        return []

    placeholders = ", ".join(f":doc_id_{index}" for index in range(len(spreadsheet_ids)))
    params = {f"doc_id_{index}": doc_id for index, doc_id in enumerate(spreadsheet_ids)}

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT c.id, c.doc_id, c.text, c.page, c.char_start, c.char_end,
                       c.section_label, c.sheet_name, c.cell_range, c.source_ref,
                       d.filename, d.file_type
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE c.doc_id IN ({placeholders})
                ORDER BY d.filename ASC, c.page ASC, c.chunk_index ASC
                """
            ),
            params,
        ).mappings().all()
    return [dict(row) for row in rows]


def tokenize(text_value):
    return set(re.findall(r"[a-z0-9]+", text_value.lower()))


def is_spreadsheet_source(filename):
    lowered = (filename or "").lower()
    return lowered.endswith(".xlsx") or lowered.endswith(".xls")


def normalize_header_label(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def extract_numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    text_value = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text_value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_sheet_metadata_line(line):
    match = re.match(
        r"\[Sheet:\s*(?P<sheet>.+?)\]\s+rows=(?P<rows>\d+),\s+cols=(?P<cols>\d+),\s+formulas=(?P<formulas>[^,]+)",
        line.strip(),
    )
    if not match:
        return None
    return {
        "sheet_name": match.group("sheet").strip(),
        "rows": int(match.group("rows")),
        "cols": int(match.group("cols")),
        "formulas": match.group("formulas").strip(),
    }


def parse_range_line(line):
    range_match = re.match(r"\[Range\s+([A-Z]+\d+(?::[A-Z]+\d+)?)\]\s*(.*)$", line.strip())
    if not range_match:
        return None

    cell_range = range_match.group(1).strip()
    payload = range_match.group(2).strip()
    cells = []
    for part in payload.split(" | "):
        cell_match = re.match(r"([A-Z]+\d+):\s*(.*)$", part.strip())
        if not cell_match:
            continue
        coordinate = cell_match.group(1)
        raw_value = cell_match.group(2).strip()
        if raw_value.startswith("formula "):
            value = raw_value.replace("formula ", "", 1).strip()
            is_formula = True
        else:
            value = raw_value
            is_formula = False
        cells.append(
            {
                "coordinate": coordinate,
                "column": re.match(r"[A-Z]+", coordinate).group(0),
                "row": int(re.search(r"\d+", coordinate).group(0)),
                "value": value,
                "numeric": extract_numeric(value),
                "is_formula": is_formula,
            }
        )

    return {"cell_range": cell_range, "cells": cells}


def parse_legacy_delimited_row(line, row_number):
    if " | " not in line:
        return None
    parts = [part.strip() for part in line.split(" | ")]
    if len(parts) < 2:
        return None

    cells = []
    for index, value in enumerate(parts, start=1):
        column = ""
        current = index
        while current > 0:
            current, remainder = divmod(current - 1, 26)
            column = chr(65 + remainder) + column
        coordinate = f"{column}{row_number}"
        cells.append(
            {
                "coordinate": coordinate,
                "column": column,
                "row": row_number,
                "value": value,
                "numeric": extract_numeric(value),
                "is_formula": False,
            }
        )
    cell_range = f"A{row_number}:{cells[-1]['column']}{row_number}"
    return {
        "cell_range": cell_range,
        "cells": cells,
        "raw_line": line.strip(),
    }


def looks_like_timestamp(value):
    return bool(
        re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?$", str(value).strip())
    )


def build_spreadsheet_model(document_id=None, collection_id=None):
    # Rebuild a sheet-like structure from stored chunks so spreadsheet questions can be answered deterministically.
    chunks = fetch_spreadsheet_chunks(document_id=document_id, collection_id=collection_id)
    if not chunks:
        return None

    documents = {}
    for chunk in chunks:
        doc = documents.setdefault(
            chunk["doc_id"],
            {
                "doc_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "file_type": chunk["file_type"],
                "sheets": defaultdict(lambda: {"metadata": {}, "rows": [], "header_map": {}}),
            },
        )

        lines = [line.strip() for line in chunk["text"].splitlines() if line.strip()]
        if not lines:
            continue

        metadata = parse_sheet_metadata_line(lines[0])
        if metadata:
            sheet_name = metadata["sheet_name"]
            sheet = doc["sheets"][sheet_name]
            sheet["metadata"].update(metadata)
            row_lines = lines[1:] if len(lines) > 1 else []
        else:
            if chunk.get("sheet_name"):
                sheet_name = chunk["sheet_name"]
            elif len(doc["sheets"]) == 1:
                sheet_name = next(iter(doc["sheets"].keys()))
            else:
                sheet_name = "Sheet 1"
            sheet = doc["sheets"][sheet_name]
            row_lines = lines

        next_legacy_row_number = max(
            [existing["row_number"] or 0 for existing in sheet["rows"]] or [0]
        ) + 1
        for row_line in row_lines:
            parsed_row = parse_range_line(row_line)
            if not parsed_row:
                parsed_row = parse_legacy_delimited_row(row_line, next_legacy_row_number)
                if parsed_row:
                    next_legacy_row_number += 1
            if not parsed_row:
                continue

            row_numbers = sorted({cell["row"] for cell in parsed_row["cells"]})
            row_number = row_numbers[0] if row_numbers else None
            row_entry = {
                "row_number": row_number,
                "cell_range": parsed_row["cell_range"],
                "cells": parsed_row["cells"],
                "chunk": chunk,
                "raw_line": parsed_row.get("raw_line"),
            }
            sheet["rows"].append(row_entry)

    for doc in documents.values():
        for sheet_name, sheet in doc["sheets"].items():
            sheet["rows"].sort(key=lambda row: (row["row_number"] or 10**9, row["cell_range"]))
            header_row = next((row for row in sheet["rows"] if row["row_number"] == 1), None)
            if header_row and not (
                header_row.get("cells") and looks_like_timestamp(header_row["cells"][0]["value"])
            ):
                header_map = {}
                for cell in header_row["cells"]:
                    header_map[cell["column"]] = str(cell["value"]).strip()
                sheet["header_map"] = header_map
            else:
                timestamp_rows = [
                    row
                    for row in sheet["rows"]
                    if row["cells"] and looks_like_timestamp(row["cells"][0]["value"])
                ]
                if timestamp_rows:
                    deduped_rows = []
                    seen_signatures = set()
                    for row in timestamp_rows:
                        signature = str(row["cells"][0]["value"]).strip()
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        deduped_rows.append(row)
                    sheet["rows"] = deduped_rows
                    sheet["implicit_header"] = True
                    if not sheet["metadata"].get("rows"):
                        sheet["metadata"]["rows"] = len(deduped_rows) + 1
                    if not sheet["metadata"].get("cols"):
                        sheet["metadata"]["cols"] = max(
                            (len(row["cells"]) for row in deduped_rows),
                            default=0,
                        )
                else:
                    deduped_rows = []
                    seen_signatures = set()
                    for row in sheet["rows"]:
                        signature = row.get("raw_line") or " | ".join(
                            str(cell["value"]).strip() for cell in row["cells"]
                        )
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        deduped_rows.append(row)
                    sheet["rows"] = deduped_rows
                    sheet["implicit_header"] = False
                    if not sheet["metadata"].get("rows"):
                        sheet["metadata"]["rows"] = len(deduped_rows)
                    if not sheet["metadata"].get("cols"):
                        sheet["metadata"]["cols"] = max(
                            (len(row["cells"]) for row in deduped_rows),
                            default=0,
                        )

    return documents


def select_relevant_sheet(model):
    ranked = []
    for doc in model.values():
        for sheet_name, sheet in doc["sheets"].items():
            row_count = sheet["metadata"].get("rows", len(sheet["rows"]))
            ranked.append((row_count, len(sheet["rows"]), doc, sheet_name, sheet))
    if not ranked:
        return None
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    _, _, doc, sheet_name, sheet = ranked[0]
    return doc, sheet_name, sheet


def infer_data_row_count(sheet):
    metadata_rows = sheet["metadata"].get("rows")
    header_offset = 1 if sheet.get("header_map") else 0
    if sheet.get("implicit_header"):
        return len(sheet["rows"])
    if metadata_rows:
        return max(metadata_rows - header_offset, 0)
    if not sheet.get("header_map"):
        return len(sheet["rows"])
    data_rows = [row for row in sheet["rows"] if row["row_number"] and row["row_number"] > 1]
    return len(data_rows)


def find_matching_columns(question, sheet):
    question_label = normalize_header_label(question)
    matches = []
    for column, header in sheet.get("header_map", {}).items():
        normalized_header = normalize_header_label(header)
        if not normalized_header:
            continue
        header_tokens = set(normalized_header.split())
        question_tokens = set(question_label.split())
        overlap = len(header_tokens.intersection(question_tokens))
        if normalized_header in question_label:
            overlap += 3
        if overlap > 0:
            matches.append((overlap, column, header))
    matches.sort(reverse=True)
    return matches


def get_column_values(sheet, column):
    values = []
    for row in sheet["rows"]:
        if row["row_number"] == 1:
            continue
        for cell in row["cells"]:
            if cell["column"] == column:
                values.append(cell)
    return values


def build_deterministic_source(doc, sheet_name, sheet, row_entry=None):
    if row_entry:
        source_ref = f"{sheet_name}!{row_entry['cell_range']}"
        snippet = row_entry["chunk"]["text"][:400]
        page = row_entry["chunk"]["page"]
        char_start = row_entry["chunk"]["char_start"]
        char_end = row_entry["chunk"]["char_end"]
        chunk_id = row_entry["chunk"]["id"]
    else:
        source_ref = f"{sheet_name}"
        snippet = f"[Sheet: {sheet_name}] rows={sheet['metadata'].get('rows', len(sheet['rows']))}, cols={sheet['metadata'].get('cols', 0)}"
        page = 1
        char_start = 0
        char_end = len(snippet)
        chunk_id = f"deterministic-{doc['doc_id']}-{sheet_name}"

    return {
        "id": chunk_id,
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "page": page,
        "section_label": "sheet-calculation",
        "sheet_name": sheet_name,
        "cell_range": row_entry["cell_range"] if row_entry else None,
        "source_ref": source_ref,
        "char_start": char_start,
        "char_end": char_end,
        "score": 1.0,
        "semantic_score": 1.0,
        "lexical_score": 1.0,
        "text": snippet,
    }


def answer_spreadsheet_question_deterministically(question, document_id=None, collection_id=None):
    # Prefer explicit spreadsheet calculations for counts and dimensions before asking the LLM to summarize.
    model = build_spreadsheet_model(document_id=document_id, collection_id=collection_id)
    if not model:
        return None

    selected = select_relevant_sheet(model)
    if not selected:
        return None
    doc, sheet_name, sheet = selected
    question_lower = question.lower()

    if (
        (re.search(r"\b(rows)\b", question_lower) and re.search(r"\b(columns|cols)\b", question_lower))
        or re.search(r"\b(rows?\s+and\s+(columns|cols)|(columns|cols)\s+and\s+rows?)\b", question_lower)
    ):
        total_rows = sheet["metadata"].get("rows", len(sheet["rows"]))
        total_columns = sheet["metadata"].get("cols", max((len(row["cells"]) for row in sheet["rows"]), default=0))
        return {
            "answer": f'The spreadsheet "{sheet_name}" has {total_rows} rows and {total_columns} columns.',
            "sources": [build_deterministic_source(doc, sheet_name, sheet)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    if re.search(r"\b(how many|number of|count)\b", question_lower) and re.search(
        r"\b(responses|response|rows|entries|records|submissions)\b",
        question_lower,
    ):
        count = infer_data_row_count(sheet)
        return {
            "answer": f'The form collected {count} responses in "{sheet_name}".',
            "sources": [build_deterministic_source(doc, sheet_name, sheet)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    column_matches = find_matching_columns(question, sheet)
    if not column_matches:
        return None

    _, best_column, best_header = column_matches[0]
    column_cells = get_column_values(sheet, best_column)
    if not column_cells:
        return None

    numeric_cells = [cell for cell in column_cells if cell["numeric"] is not None]

    if re.search(r"\b(average|avg|mean)\b", question_lower) and numeric_cells:
        value = mean(cell["numeric"] for cell in numeric_cells)
        source = next((row for row in sheet["rows"] if row["row_number"] and row["row_number"] > 1 and any(cell["column"] == best_column for cell in row["cells"])), None)
        return {
            "answer": f'The average for "{best_header}" in "{sheet_name}" is {value:.2f}.',
            "sources": [build_deterministic_source(doc, sheet_name, sheet, source)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    if re.search(r"\b(sum|total)\b", question_lower) and numeric_cells:
        value = sum(cell["numeric"] for cell in numeric_cells)
        source = next((row for row in sheet["rows"] if row["row_number"] and row["row_number"] > 1 and any(cell["column"] == best_column for cell in row["cells"])), None)
        return {
            "answer": f'The total for "{best_header}" in "{sheet_name}" is {value:.2f}.',
            "sources": [build_deterministic_source(doc, sheet_name, sheet, source)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    if re.search(r"\b(max|highest|largest)\b", question_lower) and numeric_cells:
        best_cell = max(numeric_cells, key=lambda cell: cell["numeric"])
        source = next((row for row in sheet["rows"] if row["row_number"] == best_cell["row"]), None)
        return {
            "answer": f'The highest value for "{best_header}" in "{sheet_name}" is {best_cell["numeric"]:.2f} at {best_cell["coordinate"]}.',
            "sources": [build_deterministic_source(doc, sheet_name, sheet, source)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    if re.search(r"\b(min|lowest|smallest)\b", question_lower) and numeric_cells:
        best_cell = min(numeric_cells, key=lambda cell: cell["numeric"])
        source = next((row for row in sheet["rows"] if row["row_number"] == best_cell["row"]), None)
        return {
            "answer": f'The lowest value for "{best_header}" in "{sheet_name}" is {best_cell["numeric"]:.2f} at {best_cell["coordinate"]}.',
            "sources": [build_deterministic_source(doc, sheet_name, sheet, source)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    value_match = re.search(r"\b(yes|no|male|female|true|false)\b", question_lower)
    if re.search(r"\b(how many|count|number of)\b", question_lower) and value_match:
        target_value = value_match.group(1).lower()
        matching_cells = [cell for cell in column_cells if str(cell["value"]).strip().lower() == target_value]
        source = next((row for row in sheet["rows"] if row["row_number"] == matching_cells[0]["row"]), None) if matching_cells else None
        return {
            "answer": f'There are {len(matching_cells)} rows where "{best_header}" is "{target_value}" in "{sheet_name}".',
            "sources": [build_deterministic_source(doc, sheet_name, sheet, source)] if source else [build_deterministic_source(doc, sheet_name, sheet)],
            "model": "deterministic-spreadsheet",
            "fallback_used": False,
            "error": "",
        }

    return None


def keyword_overlap_score(question_tokens, chunk_text):
    if not question_tokens:
        return 0.0
    chunk_tokens = tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    overlap = question_tokens.intersection(chunk_tokens)
    return len(overlap) / len(question_tokens)


def compact_spreadsheet_text(chunk_text, question_tokens, line_limit=SPREADSHEET_LINE_LIMIT):
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not lines:
        return chunk_text[:600]

    scored_lines = []
    for index, line in enumerate(lines):
        line_tokens = tokenize(line)
        overlap = len(question_tokens.intersection(line_tokens))
        scored_lines.append((overlap, -index, line))

    scored_lines.sort(reverse=True)
    selected = [line for overlap, _, line in scored_lines[:line_limit] if overlap > 0]

    if not selected:
        selected = lines[:line_limit]

    result = " | ".join(selected)
    return result[:900]


def build_conversation_context(conversation_history):
    if not conversation_history:
        return ""

    lines = []
    for message in conversation_history[-6:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        message_text = (message.get("message") or "").strip()
        if message_text:
            lines.append(f"{role}: {message_text}")
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


def retrieve_top_chunks(question, document_id=None, collection_id=None, top_k=TOP_K):
    documents = fetch_documents(document_id=document_id, collection_id=collection_id)
    if not documents:
        return []

    question_embedding = get_embedder().encode([question], convert_to_numpy=True).astype("float32")
    question_tokens = tokenize(question)
    candidates = []
    candidates_by_doc = {}

    for document in documents:
        index, mapping = load_document_index(document["id"])
        doc_search_k = min(max(top_k, INITIAL_RETRIEVAL_K), index.ntotal)
        if doc_search_k == 0:
            continue

        distances, indices = index.search(question_embedding, doc_search_k)
        chunk_ids = [mapping[vector_index] for vector_index in indices[0] if vector_index >= 0]
        chunk_lookup = fetch_chunks(chunk_ids)

        for vector_index, distance in zip(indices[0], distances[0]):
            if vector_index < 0:
                continue

            chunk_id = mapping[vector_index]
            chunk = chunk_lookup.get(chunk_id)
            if not chunk:
                continue

            semantic_score = 1.0 / (1.0 + float(distance))
            lexical_score = keyword_overlap_score(question_tokens, chunk["text"])
            combined_score = (semantic_score * 0.75) + (lexical_score * 0.25)

            chunk["score"] = combined_score
            chunk["semantic_score"] = semantic_score
            chunk["lexical_score"] = lexical_score
            candidates.append(chunk)
            candidates_by_doc.setdefault(document["id"], []).append(chunk)

    candidates.sort(
        key=lambda item: (item["score"], item["lexical_score"], item["semantic_score"]),
        reverse=True,
    )

    if collection_id:
        required_docs = min(len(documents), 4)
        per_doc_best = []
        for document in documents:
            doc_candidates = sorted(
                candidates_by_doc.get(document["id"], []),
                key=lambda item: (item["score"], item["lexical_score"], item["semantic_score"]),
                reverse=True,
            )
            if doc_candidates:
                per_doc_best.append(doc_candidates[0])

        selected = []
        seen_ids = set()
        for chunk in sorted(
            per_doc_best,
            key=lambda item: (item["score"], item["lexical_score"], item["semantic_score"]),
            reverse=True,
        )[:required_docs]:
            if chunk["id"] not in seen_ids:
                selected.append(chunk)
                seen_ids.add(chunk["id"])

        collection_limit = max(top_k, len(selected))
        for chunk in candidates:
            if len(selected) >= collection_limit:
                break
            if chunk["id"] in seen_ids:
                continue
            selected.append(chunk)
            seen_ids.add(chunk["id"])
        return selected

    return candidates[:top_k]


def build_prompt(
    question,
    sources,
    max_context_chars=MAX_CONTEXT_CHARS,
    conversation_history=None,
    collection_mode=False,
):
    context_parts = []
    total_chars = 0
    question_tokens = tokenize(question)
    spreadsheet_mode = any(is_spreadsheet_source(source["filename"]) for source in sources)
    context_limit = SPREADSHEET_CONTEXT_CHARS if spreadsheet_mode else max_context_chars

    for source in sources:
        page_label = source["page"] if source["page"] else "n/a"
        source_text = source["text"]
        if is_spreadsheet_source(source["filename"]):
            source_text = compact_spreadsheet_text(source_text, question_tokens)

        source_ref = source.get("source_ref") or f"Page {page_label}"
        if source.get("sheet_name") and source.get("cell_range"):
            source_ref = f"{source['sheet_name']} {source['cell_range']}"

        chunk_block = (
            f"Source: {source['filename']} | Ref: {source_ref} | Page: {page_label} | "
            f"Chars: {source['char_start']}-{source['char_end']}\n{source_text}"
        )
        if total_chars + len(chunk_block) > context_limit:
            break
        context_parts.append(chunk_block)
        total_chars += len(chunk_block)

    context = "\n\n".join(context_parts)
    spreadsheet_instruction = ""
    if spreadsheet_mode:
        spreadsheet_instruction = (
            "The context may come from spreadsheet rows or survey responses. "
            "Summarize the overall topic first, then mention only the most relevant patterns for the question. "
            "Do not repeat raw row data unless it directly supports the answer.\n\n"
        )
    collection_instruction = ""
    if collection_mode:
        collection_instruction = (
            "You are answering across multiple documents. "
            "First identify what each document is about from the provided context. "
            "Only claim a shared theme, similarity, or overlap if there is clear evidence from more than one document. "
            "If the documents are about different topics or the overlap is weak, say that clearly instead of forcing a similarity. "
            "When useful, briefly compare the documents separately before concluding.\n\n"
        )
    conversation_context = build_conversation_context(conversation_history)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{spreadsheet_instruction}"
        f"{collection_instruction}"
        f"{conversation_context}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in a direct, concise paragraph first. If the document looks like a spreadsheet or survey data, summarize the overall topic before describing notable patterns. "
        "When citing evidence, prefer the most precise source reference available, such as sheet ranges, paragraph references, or table references."
    )


def build_formatted_sources(sources):
    return [
        {
            "chunk_id": source["id"],
            "document_id": source["doc_id"],
            "filename": source["filename"],
            "page": source["page"],
            "section_label": source.get("section_label"),
            "sheet_name": source.get("sheet_name"),
            "cell_range": source.get("cell_range"),
            "source_ref": source.get("source_ref"),
            "char_start": source["char_start"],
            "char_end": source["char_end"],
            "score": round(float(source["score"]), 4),
            "semantic_score": round(float(source["semantic_score"]), 4),
            "lexical_score": round(float(source["lexical_score"]), 4),
            "snippet": source["text"][:400],
        }
        for source in sources
    ]


def fetch_available_models():
    response = requests.get(
        OLLAMA_TAGS_URL,
        timeout=(OLLAMA_CONNECT_TIMEOUT, 30),
    )
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    names = [model.get("name") for model in models if model.get("name")]
    if not names:
        return [LLM_MODEL]
    return names


def resolve_primary_model(requested_model=None):
    return requested_model or LLM_MODEL


def resolve_fallback_model(primary_model, available_models=None):
    models = available_models or fetch_available_models()
    if primary_model == FALLBACK_MODEL:
        return None
    if FALLBACK_MODEL in models:
        return FALLBACK_MODEL
    return None


def ask_ollama(prompt, model=None, timeout=OLLAMA_TIMEOUT):
    payload = {
        "model": model or LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=(OLLAMA_CONNECT_TIMEOUT, timeout),
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def answer_question(
    question,
    document_id=None,
    collection_id=None,
    top_k=TOP_K,
    model=None,
    conversation_history=None,
):
    # Spreadsheet math questions bypass normal semantic retrieval so numeric answers do not drift by model output.
    deterministic_result = answer_spreadsheet_question_deterministically(
        question,
        document_id=document_id,
        collection_id=collection_id,
    )
    if deterministic_result:
        deterministic_result["prompt"] = ""
        return deterministic_result

    sources = retrieve_top_chunks(
        question,
        document_id=document_id,
        collection_id=collection_id,
        top_k=top_k,
    )
    if not sources:
        return {
            "answer": "",
            "sources": [],
            "prompt": "",
            "error": "No indexed content matched the question.",
            "model": model or LLM_MODEL,
        }

    prompt = build_prompt(
        question,
        sources,
        conversation_history=conversation_history,
        collection_mode=bool(collection_id),
    )
    primary_model = resolve_primary_model(model)
    available_models = fetch_available_models()
    fallback_model = resolve_fallback_model(primary_model, available_models)

    try:
        answer = ask_ollama(prompt, model=primary_model, timeout=PRIMARY_MODEL_TIMEOUT)
        used_model = primary_model
        fallback_used = False
    except requests.RequestException as primary_exc:
        if not fallback_model:
            return {
                "answer": "",
                "sources": sources,
                "prompt": prompt,
                "error": f"Ollama request failed: {primary_exc}",
                "model": primary_model,
                "fallback_used": False,
            }
        try:
            answer = ask_ollama(prompt, model=fallback_model, timeout=FALLBACK_MODEL_TIMEOUT)
            used_model = fallback_model
            fallback_used = True
        except requests.RequestException as fallback_exc:
            return {
                "answer": "",
                "sources": sources,
                "prompt": prompt,
                "error": f"Llama timed out and fallback model also failed: {fallback_exc}",
                "model": fallback_model,
                "fallback_used": True,
            }
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "answer": "",
            "sources": sources,
            "prompt": prompt,
            "error": f"Ollama returned an invalid response: {exc}",
            "model": primary_model,
            "fallback_used": False,
        }

    return {
        "answer": answer,
        "sources": sources,
        "prompt": prompt,
        "error": "",
        "model": used_model,
        "fallback_used": fallback_used,
    }


def stream_answer_question(
    question,
    document_id=None,
    collection_id=None,
    top_k=TOP_K,
    model=None,
    conversation_history=None,
):
    selected_model = resolve_primary_model(model)
    deterministic_result = answer_spreadsheet_question_deterministically(
        question,
        document_id=document_id,
        collection_id=collection_id,
    )
    if deterministic_result:
        formatted_sources = build_formatted_sources(deterministic_result["sources"])
        yield {
            "type": "meta",
            "model": deterministic_result["model"],
            "sources": formatted_sources,
        }
        yield {
            "type": "done",
            "answer": deterministic_result["answer"],
            "sources": formatted_sources,
            "model": deterministic_result["model"],
            "fallback_used": False,
        }
        return

    sources = retrieve_top_chunks(
        question,
        document_id=document_id,
        collection_id=collection_id,
        top_k=top_k,
    )
    if not sources:
        yield {
            "type": "error",
            "error": "No indexed content matched the question.",
            "sources": [],
            "model": selected_model,
        }
        return

    formatted_sources = build_formatted_sources(sources)
    prompt = build_prompt(
        question,
        sources,
        conversation_history=conversation_history,
        collection_mode=bool(collection_id),
    )
    available_models = fetch_available_models()
    fallback_model = resolve_fallback_model(selected_model, available_models)

    yield {
        "type": "meta",
        "model": selected_model,
        "sources": formatted_sources,
    }
    yield {
        "type": "status",
        "stage": "generating",
        "message": f"Generating answer with {selected_model}.",
    }

    try:
        answer = ask_ollama(prompt, model=selected_model, timeout=PRIMARY_MODEL_TIMEOUT)
        yield {
            "type": "done",
            "answer": answer,
            "sources": formatted_sources,
            "model": selected_model,
            "fallback_used": False,
        }
    except requests.RequestException:
        if not fallback_model:
            yield {
                "type": "error",
                "error": f"Ollama request failed for {selected_model}.",
                "sources": formatted_sources,
                "model": selected_model,
            }
            return

        yield {
            "type": "status",
            "stage": "fallback",
            "message": f"{selected_model} is too slow right now. Switching to {fallback_model}.",
        }
        try:
            answer = ask_ollama(prompt, model=fallback_model, timeout=FALLBACK_MODEL_TIMEOUT)
            yield {
                "type": "done",
                "answer": answer,
                "sources": formatted_sources,
                "model": fallback_model,
                "fallback_used": True,
            }
        except requests.RequestException as fallback_exc:
            yield {
                "type": "error",
                "error": f"Llama timed out and fallback model also failed: {fallback_exc}",
                "sources": formatted_sources,
                "model": fallback_model,
            }
    except (ValueError, json.JSONDecodeError) as exc:
        yield {
            "type": "error",
            "error": f"Ollama returned an invalid response: {exc}",
            "sources": formatted_sources,
            "model": selected_model,
        }
