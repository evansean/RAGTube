import streamlit as st
from utils.loader import load_youtube_transcript
from utils.vector_store import create_vectorstore
from utils.rag_chain import summarize_video, run_rag_query

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAGtube 🎥", layout="wide")

# --- INITIAL SESSION STATE ---
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR ---
st.sidebar.header("🎬 RAGtube Settings")
url_input = st.sidebar.text_input("YouTube Video URL", placeholder="Paste a YouTube link...")
load_btn = st.sidebar.button("Load Video")

if load_btn and url_input:
    with st.spinner("Fetching transcript and building vector store..."):
        docs = load_youtube_transcript(url_input)
        vectordb = create_vectorstore(docs)
        summary = summarize_video(vectordb)
        st.session_state.vectordb = vectordb
        st.session_state.video_url = url_input
        st.session_state.summary = summary
        st.session_state.chat_history = []
    st.sidebar.success("Video loaded successfully!")

# --- MAIN LAYOUT ---
st.title("🎥 RAGtube — Chat with YouTube Videos")

if st.session_state.video_url:
    # Layout: video left, chat right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.video(st.session_state.video_url)
        st.markdown("### 📝 Video Summary")
        st.write(st.session_state.summary)

    with col2:
        st.markdown("### 💬 Chat about the video")
        user_query = st.text_input("Ask a question:", key="user_query")

        if user_query and st.session_state.vectordb:
            with st.spinner("Thinking..."):
                response = run_rag_query(st.session_state.vectordb, user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI", response))
            st.text_input("Ask a question:", value="", key="reset_input")  # Clear input

        # Display chat history
        for sender, msg in reversed(st.session_state.chat_history):
            if sender == "You":
                st.markdown(f"🧑 **{sender}:** {msg}")
            else:
                st.markdown(f"🤖 **{sender}:** {msg}")

else:
    st.info("👈 Enter a YouTube URL in the sidebar to begin.")
