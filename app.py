# app.py
import os
import sys
import argparse
from typing import List
from pdf_utils import pdf_pages_text, pdf_to_images
from ocr_utils import image_to_text
from chunking import chunk_page_text
from embeddings_index import EmbeddingsIndex
from genai_client import create_client, ask_gemini

FAISS_PATH = "rag.index"
PASSAGES_PATH = "passages.json"

def ingest_files(paths: List[str], use_selectable_text=True, max_tokens_per_chunk=800, overlap_tokens=150):
    chunks_meta = []
    for p in paths:
        p = os.path.abspath(p)
        if not os.path.exists(p):
            print(f"[WARN] {p} not found, skipping.")
            continue
        name = os.path.basename(p)
        ext = os.path.splitext(name)[1].lower()
        if ext == ".pdf":
            pages = pdf_pages_text(p, use_selectable_first=use_selectable_text)
            # pages is list of (page_no, text) where text may be empty if non-selectable
            # if empty, convert that page to image and OCR it
            images = None
            for page_no, txt in pages:
                if txt and txt.strip():
                    meta = chunk_page_text(txt, source=name, page_no=page_no,
                                           max_tokens=max_tokens_per_chunk, overlap_tokens=overlap_tokens)
                    chunks_meta.extend(meta)
                else:
                    # lazily convert PDF to images once if needed
                    if images is None:
                        images = pdf_to_images(p)
                    img = images[page_no - 1]
                    ocr_text = image_to_text(img)
                    meta = chunk_page_text(ocr_text, source=name, page_no=page_no,
                                           max_tokens=max_tokens_per_chunk, overlap_tokens=overlap_tokens)
                    chunks_meta.extend(meta)
        else:
            # treat as image
            from PIL import Image
            img = Image.open(p)
            txt = image_to_text(img)
            meta = chunk_page_text(txt, source=name, page_no=1,
                                   max_tokens=max_tokens_per_chunk, overlap_tokens=overlap_tokens)
            chunks_meta.extend(meta)
    return chunks_meta

def build_index_from_chunks(chunks_meta):
    passages = [c["label"] for c in chunks_meta]
    emb = EmbeddingsIndex()
    print(f"[INFO] Building embeddings for {len(passages)} passages...")
    emb.build(passages)
    emb.save(FAISS_PATH, PASSAGES_PATH)
    print(f"[INFO] Saved index to {FAISS_PATH} and passages to {PASSAGES_PATH}")
    return emb

def load_index():
    emb = EmbeddingsIndex()
    emb.load(FAISS_PATH, PASSAGES_PATH)
    print("[INFO] Loaded index.")
    return emb

def interactive_query_loop(client, emb):
    try:
        while True:
            q = input("\nEnter question (or 'quit'): ").strip()
            if not q or q.lower() in ("quit", "exit"):
                break
            retrieved = emb.query(q, top_k=4)
            if not retrieved:
                print("[INFO] No relevant passages found.")
                continue
            print("\nTop retrieved passages:")
            for r in retrieved:
                excerpt = r["passage"][:300].replace("\n", " ")
                print(f" - id={r['id']} score={r['score']:.4f} excerpt={excerpt}...")
            top_texts = [r["passage"] for r in retrieved]
            print("\n[INFO] Asking Gemini...")
            ans = ask_gemini(client, q, top_texts)
            print("\n[ANSWER]\n", ans)
    except KeyboardInterrupt:
        print("\nExiting.")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build = sub.add_parser("build")
    build.add_argument("paths", nargs="+")
    build.add_argument("--no-selectable", action="store_true")
    query = sub.add_parser("query")
    args = parser.parse_args()

    client = create_client()

    if args.cmd == "build":
        paths = args.paths
        chunks = ingest_files(paths, use_selectable_text=not args.no_selectable)
        if not chunks:
            print("[ERROR] No chunks created.")
            return
        emb = build_index_from_chunks(chunks)
    elif args.cmd == "query":
        emb = load_index()
    else:
        parser.print_help()
        return

    interactive_query_loop(client, emb)

if __name__ == "__main__":
    main()
