from langchain_community.llms import ollama
from langchain_classic.chains import RetrievalQA 

def summarize_video(vectordb):
    """Summarize the content of a video using RAG."""
    # Initialize Ollama LLM
    llm = ollama.Ollama(model="llama3", base_url="http://localhost:11434")

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=False,
    )

    # Define summarization prompt
    summarization_prompt = (
        "Provide a concise summary of the following content:\n\n{context}\n\nSummary:"
    )

    # Run the summarization
    summary = qa_chain.invoke(summarization_prompt)
    return summary

def run_rag_query(vectordb, query):
    """Run a RAG query against the vector store."""
    # Initialize Ollama LLM
    llm = ollama.Ollama(model="llama3", base_url="http://localhost:11434")

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=False,
    )

    # Run the query
    response = qa_chain.invoke(query)
    return response