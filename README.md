# 🎥 RAGtube — Chat with YouTube Videos

<img width="1919" height="975" alt="image" src="https://github.com/user-attachments/assets/12261a86-5312-4d70-8366-aaec5800eb40" />

**RAGtube** is an interactive video assistant that allows users to **ask questions about YouTube videos** and receive accurate, conversational answers. Unlike traditional video summaries or static transcripts, RAGtube combines **retrieval-augmented generation (RAG)** with **multi-turn conversational memory** to provide context-aware responses. It can answer questions based on the video content, recall details from prior conversation turns, and even respond to general knowledge questions that go beyond the video.

## Core Features

1. **YouTube Video Transcript Processing**
   - Fetches and parses YouTube video transcripts.
   - Splits transcripts into manageable chunks suitable for vector embedding.

2. **Vector Store & Retrieval**
   - Implements a **Chroma vector database** to store document embeddings.
   - Uses **MMR (Maximal Marginal Relevance)** for retrieval:
     - MMR ensures that the retrieved documents are both relevant and diverse, reducing redundancy in answers.
     - The retriever fetches the most pertinent chunks of the transcript in response to a query.

3. **RAG — Retrieval-Augmented Generation**
   - Uses a **language model (Ollama’s LLaMA 3)** to generate answers based on retrieved documents.
   - Combines **video context** with **conversation history** for coherent, multi-turn dialogues.

4. **Conversational Memory**
   - Maintains **conversation summaries** so that the assistant can remember details from previous questions and answers.
   - Supports multi-turn interaction without losing context.

5. **General Knowledge Integration**
   - The assistant can also answer questions beyond the video content, leveraging the LLM’s inherent knowledge.

6. **Streamlit Frontend**
   - Provides an **interactive chat interface** alongside the video and its summary.
   - Chat window is scrollable, while the video and summary remain static.
   - Users can ask questions in a natural, conversational manner.

## Implementation Details

- **Large Language Model:**  
  `ChatOllama` (LLaMA 3) via the Ollama LLM API.

- **Memory:**  
  Implemented using `ConversationSummaryMemory` to store and summarize prior conversation turns.

- **Vector Store:**  
  Stores embeddings of the video transcript for retrieval using `Chroma DB`. Supports **MMR** and **cosine similarity search**.

- **Prompting:**  
  - Uses a **ChatPromptTemplate** to guide the LLM in combining conversation history, video context, and user queries.
  - Responses are natural and conversational while grounded in the retrieved video content.

- **Frontend:**  
  - Built with **Streamlit**.
  - Chat window is scrollable with fixed height.
  - Video summary is displayed alongside the interactive chat.
