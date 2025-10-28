import streamlit as st
from utils.loader import load_youtube_transcript
from utils.vector_store import create_vectorstore
from utils.rag_chain import summarize_video, run_rag_query
from langchain_ollama.llms import OllamaLLM
from utils.rag_chain import create_memory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

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

# Initialize memory once
if "memory" not in st.session_state:
    llm = OllamaLLM(model="llama3")
#     llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1-0528",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     provider="auto",  # let Hugging Face choose the best provider for you
# )
#     llm = ChatHuggingFace(llm=llm)
    st.session_state.memory = create_memory(llm)

# --- SIDEBAR ---
st.sidebar.header("🎬 RAGtube Settings")
url_input = st.sidebar.text_input("YouTube Video URL", placeholder="Paste a YouTube link...")
load_btn = st.sidebar.button("Load Video")

if load_btn and url_input:
    with st.spinner("Fetching transcript..."):
        docs = load_youtube_transcript(url_input)
    st.success("Transcript fetched.")

    with st.spinner("Building vector store..."):
        vectordb = create_vectorstore(docs)
    st.success("Vector store built.")

    with st.spinner("Summarizing video..."):
        # summary = summarize_video(docs)
        print("Summarizing video...")
    st.success("Video summarized.")

    st.session_state.vectordb = vectordb
    st.session_state.video_url = url_input
    st.session_state.summary = "summary"
    st.session_state.chat_history = []
    st.session_state.memory.clear()  # Reset memory when new video is loaded
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
                response = run_rag_query(
                    st.session_state.vectordb,
                    user_query,
                    st.session_state.memory
                )
            # Append to chat history for display
            st.session_state.chat_history.append({"user": user_query, "assistant": response})
            st.text_input("Ask a question:", value="", key="reset_input")  # Clear input

        # Display chat history in reverse
        for turn in reversed(st.session_state.chat_history):
            st.markdown(f"🧑 **You:** {turn['user']}")
            st.markdown(f"🤖 **AI:** {turn['assistant']}")

else:
    st.info("👈 Enter a YouTube URL in the sidebar to begin.")
