from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from llm_client import OllamaGenerator
from haystack import Pipeline
from haystack import Document
import json

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-small-v2")
embedder.warm_up()

#Reads and saves json to a new Documents list
Embedded_List = []
with open("./data/x.jsonl", "r") as f:
    Embedded_List = [Document.from_dict(json.loads(line)) for line in f]

# Write the chunks with their embedding vector into the document_store
document_store = InMemoryDocumentStore()
document_store.write_documents(Embedded_List)

#Defining the retriever: Defines where my chunks are located and what embedding model was used. Retriver will use the same model as defined earlier
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

#Initiate the LLM
generator = OllamaGenerator(model="phi3:mini", base_url="http://localhost:11434")

#Initiate the pipeline
pipe = Pipeline()
pipe.add_component("generator", generator)

def prompt_builder(q: str):
    Context = "Answer the question using only the context provided.\n Context:\n"
    EQ = Document(content=f"query: {q}")
    temp = embedder.run(EQ.content)
    EQ.embedding = temp["embedding"]
    Context_docs = retriever.run(query_embedding=EQ.embedding, top_k=3)["documents"]
    for docs in Context_docs:
        Context += docs.content + "\n"

    Context += f"\n Question: {q}"
    return Context

# === Interactive loop ===
if __name__ == "__main__":
    while True:
        #question = "What is arbitration? use 10 words" 
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() in ["exit", "quit"]:
            break
        t = prompt_builder(question)
        response = pipe.run(data={"generator":{"prompt": t}})
        print("Response:", response["generator"]["replies"][0])