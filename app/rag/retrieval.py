from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever


def get_retriever(splits, bm25_k, mmr_k, mmr_fetch_k, metadata_field_info, document_content_description):
    llm = OpenAI(temperature=0)
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = bm25_k

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={'k': mmr_k, 'fetch_k': mmr_fetch_k}
    )
    
    self_retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, verbose=True
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[self_retriever, bm25_retriever, mmr_retriever], weights=[0.6, 0.2, 0.2]
    )

    return ensemble_retriever
