from langchain_ollama.llms import OllamaLLM
from langchain_classic.chains import LLMChain 
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

def create_memory(llm):
    """Initialize conversation summary memory."""
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        return_messages=False  # we just want summary text
    )

def summarize_video(docs):
    """Summarize the content of a video using RAG."""
    # Initialize Ollama LLM
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


    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

    # Run the summarization
    summary = chain.invoke(docs)
    return summary['output_text']

def run_rag_query(vectordb, query, memory):
    """Run a RAG query against the vector store."""
    # Initialize Ollama LLM
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


    # Retrieve relevant documents
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    print(dir(retriever))

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    template = """You are a helpful video assistant that answers questions based on a video's transcript.
        Use the transcript context and the conversation summary below to give an accurate answer.
        If the answer isn't in the transcript, say "I don't know."

        Conversation summary:
        {chat_history}

        Transcript context:
        {context}

        User question:
        {question}

        Answer:
    """
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template,
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
    )

    response = chain.invoke({"question": query, "context": context})
    return response