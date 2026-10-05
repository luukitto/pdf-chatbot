"""Web UI.  Run:  streamlit run app.py"""
import streamlit as st

from rag import answer, format_sources, get_collection, retrieve

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Runs 100% locally: llama3.2 + nomic-embed-text via Ollama, Chroma for vectors.")

pdf = st.file_uploader("Upload a PDF", type="pdf")
if not pdf:
    st.stop()

with st.spinner("Reading and embedding the PDF (only the first time)..."):
    col = get_collection(pdf.getvalue())

history = st.session_state.setdefault(col.name, [])  # separate chat per PDF


def show(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            with st.expander(format_sources(msg["hits"])):
                for h in msg["hits"]:
                    st.markdown(f"**Page {h['page']}** · similarity {h['score']:.2f}\n\n> {h['text']}")


for msg in history:
    show(msg)

if question := st.chat_input("Ask something about the document"):
    history.append({"role": "user", "content": question})
    show(history[-1])
    with st.chat_message("assistant"):
        hits = retrieve(question, col)
        reply = st.write_stream(answer(question, hits))
    history.append({"role": "assistant", "content": reply, "hits": hits})
    st.rerun()  # redraw so the new answer gets its sources expander
