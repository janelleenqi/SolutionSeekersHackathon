# from traceability import init_traceability_db

"""Run with: python -m streamlit run app.py."""
import json
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from src.chat_service import backend_ready, load_chunks, respond

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)
st.set_page_config(page_title='Wealth Advisor Assistant', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1120px; padding-top: 2.5rem;}
</style>''', unsafe_allow_html=True)
st.session_state.setdefault('messages', [])
try:
    chunks = load_chunks(ROOT / 'data/processed/chunks.jsonl')
except (OSError, ValueError) as exc:
    st.error(f'Document library unavailable: {exc}')
    chunks = []
types = sorted({c['metadata'].get('document_type', 'document') for c in chunks})
ready = backend_ready()
with st.sidebar:
    st.markdown('### ◈ Wealth workspace')
    st.caption('SOLUTION SEEKERS · APAC')
    if st.button('New conversation', use_container_width=True):
        st.session_state.messages = []
        for key in list(st.session_state):
            if key.startswith('feedback_'):
                del st.session_state[key]
        st.rerun()
    st.divider()
    st.markdown('#### Research settings')
    top_k = st.slider('Document passages to retrieve', 1, 10, 5)
    selected_types = st.multiselect('Document types', types, default=types)
    st.caption('Filters apply to documents. Named-client questions also include structured portfolio evidence.')
    if ready:
        st.success('Answer generation configured')
    else:
        st.info('Set LLM_MODEL, LLM_API_KEY and LLM_BASE_URL in the terminal or local .env file to generate answers.')
    st.caption('Ask each question in full, including the client name or ID. Conversation history is displayed but is not sent to the model.')

st.caption('RESEARCH · RELATIONSHIPS · EVIDENCE')
st.title('Wealth Advisor Assistant')
st.write('Prepare for client conversations with portfolio facts and evidence from your supplied documents.')
scoped = [c for c in chunks if c['metadata'].get('document_type', 'document') in selected_types]
left, middle, right = st.columns(3)
left.metric('Documents in scope', len({c['metadata'].get('source_path') for c in scoped}))
middle.metric('Prepared passages', len(scoped))
right.metric('Mode', 'Hybrid RAG')
st.caption('Passage counts describe the ingested library. Rebuild the Chroma index after changing source documents.')

def render_message(message, index):
    with st.chat_message(message['role']):
        if message.get('mode') == 'Error':
            st.error(message['content'])
        elif message.get('abstained'):
            st.warning(message['content'])
        else:
            st.markdown(message['content'])
        if message['role'] != 'assistant':
            return
        st.caption(f"{message.get('mode', 'Assistant')} · {message.get('elapsed', 0):.1f}s · {message.get('evidence_count', 0)} evidence blocks")
        if message.get('structured_backend'):
            st.caption('Structured source: ' + message['structured_backend'])
        if message.get('structured_fallback'):
            st.info('SQLite retrieval was unavailable or returned no rows; used the existing JSON/CSV queries.')
        if message.get('grounding_diagnostics'):
            with st.expander('Answer diagnostics (not factual verification)'):
                st.json(message['grounding_diagnostics'])
        if message.get('reason'):
            with st.expander('Why this answer was withheld'):
                st.write(message['reason'])
        for source in message.get('sources', []):
            metadata = source['metadata']
            locator = metadata.get('source_path', 'Source')
            if metadata.get('page') is not None:
                locator += f" · page {metadata['page']}"
            status = 'Cited' if source['cited'] else 'Retrieved'
            with st.expander(f"[{source['label']}] {locator} · {status}"):
                st.text(source['text'])
                if metadata.get('evidence_type') == 'structured':
                    st.caption('Structured records and calculations · source rows listed above')
                else:
                    st.caption(f"Similarity: {source.get('score', 0):.3f} · Chunk: {source['id']}")
        if message.get('mode') != 'Error':
            message['feedback'] = st.radio('Was this useful?', ['Not rated', 'Helpful', 'Needs improvement'], horizontal=True, key=f'feedback_{index}')

disabled = not chunks or not selected_types or not ready
if not chunks:
    st.warning('Run ingestion and indexing to prepare the document library. See the README.')
prompt = None
if not st.session_state.messages:
    st.subheader('Start with a question')
    starters = ['What is the SRI rating and minimum initial investment for the APAC Stable Income Money Market Fund?',
                'Is Robert Chua suitable for APEX?', 'What pending transactions does James Sullivan have?']
    for column, question in zip(st.columns(3), starters):
        if column.button(question, use_container_width=True, disabled=disabled):
            prompt = question
for index, message in enumerate(st.session_state.messages):
    render_message(message, index)
prompt = st.chat_input('Ask about a client, fund, or policy…', disabled=disabled) or prompt
if prompt and prompt.strip():
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    started = time.perf_counter()
    with st.spinner('Retrieving evidence and preparing your response…'):
        try:
            result = respond(prompt, top_k=top_k, document_types=selected_types, preview=False)
            message = {**result, 'role': 'assistant', 'content': result['answer']}
        except Exception as exc:
            message = {'role': 'assistant', 'content': f'Request failed: {exc}', 'mode': 'Error'}
        message['elapsed'] = time.perf_counter() - started
        st.session_state.messages.append(message)
    st.rerun()
if st.session_state.messages:
    st.sidebar.download_button('Download conversation', json.dumps(st.session_state.messages, indent=2, ensure_ascii=False), file_name='wealth-advisor-chat.json', mime='application/json', use_container_width=True)
