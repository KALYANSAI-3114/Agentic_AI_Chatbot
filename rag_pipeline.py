import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pdf_ingester import PDFIngester
from typing import List, Dict, Any
from datetime import datetime
import faiss
import numpy as np

load_dotenv()

# FAISS Vector Store for local embeddings

class FAISSVectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
    
    def add_documents(self, documents: List[str]):
        """Local FAISS embeddings"""
        self.documents = documents
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"Stored {len(documents)} chunks in FAISS")
    
    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """Retrieve relevant chunks"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        contexts = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                similarity = 1.0 - (dist / 2.0)
                contexts.append({
                    "text": self.documents[idx],
                    "score": max(0.0, similarity),
                    "source": "Agentic AI eBook"
                })
        return contexts

class SarvamAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sarvam.ai"
    
    def chat_completion(self, messages: List[Dict], model: str = "sarvam-m") -> str:
        """Sarvam AI chat completion"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Sarvam API error: {response.status_code} - {response.text}")

class RAGPipeline:
    def __init__(self, pdf_path: str = "Ebook-Agentic-AI.pdf"):
        self.pdf_path = pdf_path
        self.vector_store = FAISSVectorStore()
        self.sarvam = SarvamAPI(os.getenv("SARVAM_API_KEY"))
        self._setup()
    
    def _setup(self):
        print("Loading PDF...")
        ingester = PDFIngester()
        text = ingester.read_pdf(self.pdf_path)
        documents = ingester.chunk_text(text)
        doc_texts = [doc.page_content for doc in documents]
        
        print("Building FAISS index...")
        self.vector_store.add_documents(doc_texts)
        print("Sarvam RAG pipeline ready!")

    #Prompt Template
    def invoke(self, question: str) -> Dict[str, Any]:
        """Full RAG pipeline"""
        contexts = self.vector_store.retrieve(question)
        context_text = "\n\n".join([
            f"CHUNK {i+1} ({ctx['score']:.3f}):\n{ctx['text'][:1000]}" 
            for i, ctx in enumerate(contexts)
        ])
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert on Agentic AI. Answer STRICTLY from the eBook context below.

RULES:
1. ONLY use CONTEXT CHUNKS provided (ignore external knowledge)
2. If answer not in context: "I don't have enough information from the Agentic AI eBook."
3. Cite chunks used: [Chunk 1], [Chunk 2], etc.

CONTEXT CHUNKS:
{context_text}"""
            },
            {"role": "user", "content": question}
        ]
        
        answer = self.sarvam.chat_completion(messages, model="sarvam-m")
        confidence = sum(ctx["score"] for ctx in contexts) / max(1, len(contexts))
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
