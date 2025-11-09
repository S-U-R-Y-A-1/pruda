# streamlit_app.py (updated with Save + Download index functionality)
import os
import tempfile
import zipfile
import io
import streamlit as st
from typing import List
from dotenv import load_dotenv

# Load .env so GENAI_API_KEY is available if present
load_dotenv(dotenv_path=".env", override=False)

# import pipeline functions & constants from your app module
from app import ingest_files, build_index_from_chunks, load_index, FAISS_PATH, PASSAGES_PATH
from genai_client import create_client, ask_gemini
from embeddings_index import EmbeddingsIndex

st.set_page_config(page_title="RAG OCR → Gemini", layout="wide")

st.title("RAG: PDF/Image OCR → FAISS → Gemini")

# No API key input required — we auto-read it from .env / environment
if not os.getenv("GENAI_API_KEY"):
    st.warning(
        "GENAI_API_KEY not found. Put your key in a .env file or set the GENAI_API_KEY environment variable. "
        "Uploads & index build will be disabled until key is provided."
    )
    upload_disabled = True
else:
    upload_disabled = False

uploaded = st.file_uploader(
    "Upload PDFs or images (multiple)",
    accept_multiple_files=True,
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    disabled=upload_disabled,
)

build_btn = st.button("Build index from uploaded files", disabled=upload_disabled)
load_btn = st.button("Load existing index (from disk)")

# Save / download UI elements (only enabled when an index object exists in session)
st.sidebar.markdown("## Index file options")
index_filename = st.sidebar.text_input("Index filename", value=FAISS_PATH)
passages_filename = st.sidebar.text_input("Passages filename", value=PASSAGES_PATH)
zip_name = st.sidebar.text_input("Zip filename for download", value="rag_index.zip")

# init session state
if "index_built" not in st.session_state:
    st.session_state["index_built"] = False
    st.session_state["emb"] = None
    st.session_state["num_passages"] = 0

# Build index from uploaded files
if build_btn and uploaded:
    with st.spinner("Saving uploads and building index (this may take a while)..."):
        tmpdir = tempfile.mkdtemp(prefix="rag_upload_")
        paths = []
        for f in uploaded:
            out = os.path.join(tmpdir, f.name)
            with open(out, "wb") as wf:
                wf.write(f.getbuffer())
            paths.append(out)
        chunks_meta = ingest_files(paths)
        if not chunks_meta:
            st.error("No text found in uploads.")
        else:
            emb = build_index_from_chunks(chunks_meta)
            st.session_state["index_built"] = True
            st.session_state["emb"] = emb
            st.session_state["num_passages"] = len(emb.passages)
            st.success(f"Built index with {len(emb.passages)} passages.")

# Load existing index
if load_btn:
    try:
        emb = load_index()
        st.session_state["index_built"] = True
        st.session_state["emb"] = emb
        st.session_state["num_passages"] = len(emb.passages)
        st.success(f"Loaded index with {len(emb.passages)} passages.")
    except Exception as e:
        st.error(f"Failed to load index: {e}")

# If index is available in session, show query UI and save/download options
if st.session_state["index_built"] and st.session_state["emb"] is not None:
    st.markdown(f"**Index loaded with {st.session_state['num_passages']} passages.**")

    # Save index button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save index to disk"):
            try:
                emb: EmbeddingsIndex = st.session_state["emb"]
                emb.save(index_path=index_filename, passages_path=passages_filename)
                st.success(f"Index saved to '{index_filename}' and passages saved to '{passages_filename}'.")
            except Exception as e:
                st.error(f"Failed to save index: {e}")

    # Download index as zip
    with col2:
        if st.button("Create download ZIP"):
            try:
                # ensure files exist on disk (if they were not saved yet, save to temp paths)
                emb: EmbeddingsIndex = st.session_state["emb"]
                # If user hasn't saved yet to disk, save to temp files first
                save_index_path = index_filename
                save_passages_path = passages_filename
                # If the target filenames are not present, write to them now (overwrite)
                try:
                    emb.save(index_path=save_index_path, passages_path=save_passages_path)
                except Exception as e_save:
                    # If saving to given paths fails, use temp files
                    tmp_dir = tempfile.mkdtemp(prefix="rag_index_")
                    save_index_path = os.path.join(tmp_dir, os.path.basename(index_filename))
                    save_passages_path = os.path.join(tmp_dir, os.path.basename(passages_filename))
                    emb.save(index_path=save_index_path, passages_path=save_passages_path)

                # create in-memory zip
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(save_index_path, arcname=os.path.basename(save_index_path))
                    zipf.write(save_passages_path, arcname=os.path.basename(save_passages_path))
                zip_buffer.seek(0)

                st.download_button(
                    label="Download index (.zip)",
                    data=zip_buffer,
                    file_name=zip_name,
                    mime="application/zip",
                )
                st.success("ZIP created. Click the download button above.")
            except Exception as e:
                st.error(f"Failed to create/download ZIP: {e}")

    st.markdown("---")
    st.markdown("### Ask a question (uses the built/loaded FAISS index)")
    question = st.text_input("Question")
    top_k = st.slider("Top-k passages to retrieve", 1, 8, 4)
    if st.button("Ask Gemini") and question.strip():
        try:
            client = create_client()
        except Exception as e:
            st.error(f"GENAI client creation failed: {e}")
            client = None

        if client:
            emb: EmbeddingsIndex = st.session_state["emb"]
            retrieved = emb.query(question, top_k=top_k)
            if not retrieved:
                st.info("No relevant passages found.")
            else:
                st.write("#### Retrieved passages")
                for r in retrieved:
                    st.write(f"**score**: {r['score']:.4f} — excerpt:", r["passage"][:600].replace("\n", " "))
                top_texts = [r["passage"] for r in retrieved]
                with st.spinner("Asking Gemini..."):
                    answer = ask_gemini(client, question, top_texts)
                st.write("### Gemini answer")
                st.write(answer)
else:
    st.info("Build an index from uploads or load an existing index to start querying.")
