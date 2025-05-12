import os
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains.history_aware_retriever import create_history_aware_retriever


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    print(os.getenv("INDEX_NAME"))
    embeddings = OllamaEmbeddings(model="qwen3:latest")
    docsearch = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
    chat = ChatOllama(verbose=True, temperature=0, model="gemma3:latest")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    # return result
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])
