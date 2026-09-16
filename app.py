"""Run with: python -m streamlit run app.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import time

import streamlit as st

from src.chat_service import backend_ready, load_chunks, respond
from src.ingestion import IngestionConfig, ingest_file

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Wealth Advisor Assistant", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp {background: #f5f7fb;}
.block-container {max-width: 1120px; padding-top: 2.5rem;}
[data-testid="stSidebar"] {background: #eaf0f6;}
h1, h2, h3 {color: #142b49;}
[data-testid="stChatMessage"] {background: white; border: 1px solid #e1e7ef; border-radius: 14px;}
.eyebrow {color: #367b79; font-size: .78rem; font-weight: 700; letter-spacing: .16em;}
</style>
""", unsafe_allow_html=True)

for key, default in {"messages": [], "uploaded_chunks": [], "upload_report": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

try:
    base_chunks = load_chunks(ROOT / "data/processed/chunks.jsonl")
except (OSError, ValueError) as exc:
    st.error(f"Could not load the document library: {exc}")
    base_chunks = []

with st.sidebar:
    st.markdown("### ◈ Wealth workspace")
    st.caption("SOLUTION SEEKERS · HACKATHON")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        for key in list(st.session_state):
            if key.startswith("feedback_"):
                del st.session_state[key]
        st.rerun()
    st.divider()
    st.markdown("#### Document library")
    st.caption("Use the ingested library or add documents for this session.")
    uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md", "csv", "json"], accept_multiple_files=True)
    if st.button("Process uploaded documents", disabled=not uploads, use_container_width=True):
        collected, report = [], []
        with st.spinner("Reading and preparing passages…"):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                for index, upload in enumerate(uploads):
                    # Preserve filenames for the ingestion module's structured parsers.
                    folder = root / str(index)
                    folder.mkdir()
                    path = folder / Path(upload.name.replace("\\", "/")).name
                    try:
                        path.write_bytes(upload.getvalue())
                        config = IngestionConfig(root, root, include_structured=True)
                        new_chunks = ingest_file(path, config)
                        collected.extend(new_chunks)
                        report.append((bool(new_chunks), f"{path.name}: {len(new_chunks)} passages"))
                    except Exception as exc:
                        report.append((False, f"{path.name}: {exc}"))
        st.session_state.uploaded_chunks = collected
        st.session_state.upload_report = report
    for success, message in st.session_state.upload_report:
        (st.success if success else st.warning)(message)
    if st.session_state.uploaded_chunks and st.button("Remove session documents"):
        st.session_state.uploaded_chunks = []
        st.session_state.upload_report = []
        st.rerun()
    top_k = st.slider("Maximum source passages", 1, 8, 3)

all_chunks = base_chunks + st.session_state.uploaded_chunks
types = sorted({c["metadata"].get("document_type", "document") for c in all_chunks})
with st.sidebar:
    selected_types = st.multiselect("Document types", types, default=types)
    st.divider()
    st.caption("Backend: RAG connected" if backend_ready() else "Backend: local document search")
    st.caption("Uploads and conversation history are held in this browser session. Refreshing may reset them.")

chunks = [c for c in all_chunks if c["metadata"].get("document_type", "document") in selected_types]
st.markdown('<div class="eyebrow">RESEARCH · RELATIONSHIPS · EVIDENCE</div>', unsafe_allow_html=True)
st.title("Wealth Advisor Assistant")
st.markdown("Prepare for your next client conversation, with the source material close at hand.")
left, middle, right = st.columns(3)
left.metric("Documents in scope", len({c["metadata"].get("source_path", c["id"]) for c in chunks}))
middle.metric("Searchable passages", len(chunks))
right.metric("Assistant mode", "RAG" if backend_ready() else "Document search")
if not backend_ready():
    st.info("Document-search preview · Search real passages without an API key. Generated answers and semantic retrieval become available when your RAG backend is connected.")
if not chunks:
    st.warning("No passages in scope. Upload documents, select document types, or run the ingestion command in the README.")


def render_message(message, index):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"{message.get('mode', 'Assistant')} · {message.get('elapsed', 0):.1f}s")
            for number, source in enumerate(message.get("sources", []), 1):
                metadata = source["metadata"]
                label = metadata.get("source_path", metadata.get("source_filename", "Source"))
                locator = " · ".join(f"{key.replace('_', ' ')} {metadata[key]}" for key in ("page", "record_id", "client_id") if key in metadata)
                with st.expander(f"[{number}] {label}" + (f" · {locator}" if locator else "")):
                    st.text(source["text"])
                    st.caption(f"Passage ID: {source.get('id', 'not supplied')}")
            feedback = st.radio("Was this useful?", ["Not rated", "Helpful", "Needs improvement"], horizontal=True, key=f"feedback_{index}", index=["Not rated", "Helpful", "Needs improvement"].index(message.get("feedback", "Not rated")))
            message["feedback"] = feedback


prompt = None
if not st.session_state.messages:
    st.subheader("Start with a question")
    st.caption("Explore your supplied factsheets, client correspondence, and policy documents.")
    starters = ["What are the key risks in the APAC funds?", "Which clients have a high risk appetite?", "Find evidence about client suitability."]
    for column, question in zip(st.columns(3), starters):
        if column.button(question, use_container_width=True, disabled=not chunks):
            prompt = question

for index, message in enumerate(st.session_state.messages):
    render_message(message, index)

typed = st.chat_input("Ask about a client, fund, or policy…", disabled=not chunks)
prompt = typed or prompt
if prompt and prompt.strip():
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Searching the evidence…"):
        started = time.perf_counter()
        try:
            result = respond(prompt, chunks, history, top_k)
            message = {"role": "assistant", "content": result["answer"], "sources": result.get("sources", []), "mode": result["mode"], "abstained": result.get("abstained", False)}
        except Exception:
            message = {"role": "assistant", "content": "The assistant could not complete this request. Check the backend configuration and try again.", "mode": "Error", "sources": []}
        message["elapsed"] = time.perf_counter() - started
        st.session_state.messages.append(message)
    st.rerun()

if st.session_state.messages:
    st.sidebar.download_button("Download conversation", json.dumps(st.session_state.messages, indent=2, ensure_ascii=False), file_name="wealth-advisor-chat.json", mime="application/json", use_container_width=True)
