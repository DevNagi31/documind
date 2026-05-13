# RAG Basics

Retrieval-Augmented Generation (RAG) is a technique for grounding LLM responses
in external data. A RAG pipeline has four key stages:

1. **Ingestion** — load documents, split them into chunks, embed each chunk,
   and store the embeddings in a vector database.
2. **Retrieval** — embed the user's query and find the most similar chunks in
   the vector store. Maximum Marginal Relevance (MMR) can be used to balance
   relevance with diversity.
3. **Generation** — pass the retrieved chunks to an LLM as context, instruct it
   to answer only from that context, and require source citations.
4. **Evaluation** — score answers along multiple axes: faithfulness, answer
   relevance, context precision, and hallucination rate.

Chunking strategy matters: too-large chunks dilute relevance, too-small chunks
lose context. Recursive character splitting with ~500-token chunks and ~80-token
overlap is a sensible default for prose.
