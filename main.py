import os
import logging
import csv
from datetime import datetime
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from typing import List, Dict, Tuple
import re
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import argparse
import asyncio
import base64
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB client
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client["chatbot_db"]
user_chats_collection = db["user_chats"]

# Ensure MongoDB indexes
async def init_mongodb():
    try:
        await user_chats_collection.create_index("user_id", unique=True)
        logger.info("MongoDB: Initialized indexes")
    except Exception as e:
        logger.error(f"MongoDB-INIT-ERROR: Failed to initialize indexes: {str(e)}")

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_mongodb()
    yield
    mongo_client.close()

# FastAPI app with lifespan
app = FastAPI(title="RAG Chatbot with Voice Input and TTS", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGPipeline class (unchanged except for generate_answer)
class RAGPipeline:
    """RAG pipeline for searching FAISS index and generating answers with OpenAI LLM."""
    
    def __init__(self, faiss_index_path: str, chunk_info_path: str, model_name: str = 'all-MiniLM-L6-v2', llm_model: str = 'gpt-4o-mini'):
        """Initialize pipeline with FAISS index, chunk info, and models."""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.faiss_index_path = faiss_index_path
        self.chunk_info_path = chunk_info_path
        self.llm_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.llm_model = llm_model
        
        # Load FAISS index
        try:
            self.index = faiss.read_index(faiss_index_path)
            logger.info(f"FAISS-LOAD: Loaded index from {faiss_index_path}")
        except Exception as e:
            logger.error(f"FAISS-LOAD-ERROR: Failed to load index: {str(e)}")
            raise
        
        # Load chunk info
        try:
            with open(chunk_info_path, 'rb') as f:
                self.chunk_info = pickle.load(f)
            logger.info(f"CHUNK-INFO-LOAD: Loaded chunk info from {chunk_info_path}")
        except Exception as e:
            logger.error(f"CHUNK-INFO-LOAD-ERROR: Failed to load chunk info: {str(e)}")
            raise
        
        # Initialize BM25
        self.corpus = [info['chunk'] for info in self.chunk_info.values()]
        tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25-INIT: Initialized BM25 index")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return re.findall(r'\w+', text.lower())
    
    def _normalize_scores(self, scores: np.ndarray, min_score: float = None, max_score: float = None) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if min_score is None:
            min_score = scores.min()
        if max_score is None:
            max_score = scores.max()
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def search(self, query: str, top_k: int = 5, faiss_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Tuple[int, float]]:
        """Hybrid search combining FAISS and BM25."""
        try:
            query_embedding = self.model.encode([query], show_progress_bar=False)[0].astype(np.float32)
            logger.info(f"QUERY-EMBED: Embedded query: {query}")
        except Exception as e:
            logger.error(f"QUERY-EMBED-ERROR: Failed to embed query: {str(e)}")
            return []
        
        try:
            distances, indices = self.index.search(np.array([query_embedding]), top_k * 2)
            faiss_scores = 1 / (1 + distances[0])
            faiss_scores = self._normalize_scores(faiss_scores)
            logger.info(f"FAISS-SEARCH: Retrieved {len(indices[0])} chunks")
        except Exception as e:
            logger.error(f"FAISS-SEARCH-ERROR: Failed to search FAISS: {str(e)}")
            faiss_scores = np.zeros(top_k * 2)
            indices = [[]]
        
        try:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_scores = self._normalize_scores(bm25_scores)
            logger.info("BM25-SEARCH: Computed BM25 scores")
        except Exception as e:
            logger.error(f"BM25-SEARCH-ERROR: Failed to compute BM25 scores: {str(e)}")
            bm25_scores = np.zeros(len(self.corpus))
        
        combined_scores = {}
        for i, idx in enumerate(indices[0]):
            if idx in self.chunk_info:
                combined_scores[idx] = faiss_weight * faiss_scores[i]
        
        for i, score in enumerate(bm25_scores):
            if i in self.chunk_info:
                combined_scores[i] = combined_scores.get(i, 0) + bm25_weight * score
        
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info(f"HYBRID-SEARCH: Retrieved top {len(sorted_scores)} chunks")
        return sorted_scores
    
    async def generate_answer(self, query: str, chunk_ids: List[int], user_id: str, max_context_length: int = 4000) -> str:
        """Generate answer using OpenAI LLM with retrieved chunks and chat history."""
        context = []
        total_length = 0
        
        # Fetch recent chat history from MongoDB
        chat_history = []
        try:
            user_doc = await user_chats_collection.find_one({"user_id": user_id})
            if user_doc and "chats" in user_doc:
                chats = sorted(user_doc["chats"], key=lambda x: x["timestamp"], reverse=True)[:5]
                for chat in chats:
                    chat_text = f"User: {chat['query']}\nAssistant: {chat['answer']}\n"
                    if total_length + len(chat_text) <= max_context_length // 2:
                        chat_history.append(chat_text)
                        total_length += len(chat_text)
                    else:
                        break
            logger.info(f"CHAT-HISTORY: Retrieved {len(chat_history)} chats for user {user_id}")
        except Exception as e:
            logger.error(f"CHAT-HISTORY-ERROR: Failed to fetch chat history for user {user_id}: {str(e)}")
        
        if chat_history:
            context.append("Chat History:\n" + "\n".join(chat_history))
        
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_info:
                chunk = self.chunk_info[chunk_id]['chunk']
                metadata = self.chunk_info[chunk_id]['metadata']
                chunk_text = f"Document: {metadata['filename']}\nKeywords: {', '.join(metadata['keywords'])}\nContent: {chunk}\n"
                if total_length + len(chunk_text) <= max_context_length:
                    context.append(chunk_text)
                    total_length += len(chunk_text)
                else:
                    break
        
        context_str = "\n".join(context)
        prompt = (
            "You are an assistant helping with information retrieval from documents and user chat history. "
            "Use the provided context, which includes past user interactions and document chunks, to answer the query concisely and accurately. "
            "If the context doesn't contain relevant information, say so. "
            f"\n\nQuery: {query}\n\nContext:\n{context_str}\n\nAnswer:"
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM-GENERATE: Generated answer for query: {query}")
            return answer
        except Exception as e:
            logger.error(f"LLM-GENERATE-ERROR: Failed to generate answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def log_search(self, query: str, chunk_ids: List[int], answer: str, csv_path: str):
        """Log search details to CSV."""
        fieldnames = ['query', 'retrieved_chunk_ids', 'answer', 'timestamp', 'error']
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        
        record = {
            'query': query,
            'retrieved_chunk_ids': ','.join(map(str, chunk_ids)),
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'error': '' if 'Error' not in answer else answer
        }
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        logger.info(f"LOG: Saved search log to {csv_path}")

# Original main function (unchanged)
def main():
    config = {
        'faiss_index_path': 'output/faiss_index.bin',
        'chunk_info_path': 'output/faiss_chunk_info.pkl',
        'csv_log': 'logger/rag_log.csv',
        'top_k': 5,
        'faiss_weight': 0.7,
        'bm25_weight': 0.3
    }
    
    try:
        pipeline = RAGPipeline(
            faiss_index_path=config['faiss_index_path'],
            chunk_info_path=config['chunk_info_path']
        )
    except Exception as e:
        logger.error(f"INIT-ERROR: Failed to initialize pipeline: {str(e)}")
        return
    
    print("Enter your query (or 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            print("Please enter a valid query.")
            continue
        
        try:
            results = pipeline.search(query, top_k=config['top_k'], 
                                   faiss_weight=config['faiss_weight'], 
                                   bm25_weight=config['bm25_weight'])
            chunk_ids = [chunk_id for chunk_id, score in results]
            if not chunk_ids:
                print("No relevant chunks found.")
                pipeline.log_search(query, [], "No relevant chunks found", config['csv_log'])
                continue
        except Exception as e:
            print(f"Search error: {str(e)}")
            pipeline.log_search(query, [], f"Search error: {str(e)}", config['csv_log'])
            continue
        
        try:
            answer = asyncio.run(pipeline.generate_answer(query, chunk_ids, user_id="cli_user"))
            print("\nAnswer:")
            print(answer)
            print()
            pipeline.log_search(query, chunk_ids, answer, config['csv_log'])
        except Exception as e:
            print(f"Generation error: {str(e)}")
            pipeline.log_search(query, chunk_ids, f"Generation error: {str(e)}", config['csv_log'])

# Initialize RAG pipeline for FastAPI
config = {
    'faiss_index_path': 'output/faiss_index.bin',
    'chunk_info_path': 'output/faiss_chunk_info.pkl',
    'csv_log': 'logger/rag_log.csv',
    'top_k': 5,
    'faiss_weight': 0.7,
    'bm25_weight': 0.3
}

try:
    pipeline = RAGPipeline(
        faiss_index_path=config['faiss_index_path'],
        chunk_info_path=config['chunk_info_path']
    )
except Exception as e:
    logger.error(f"INIT-ERROR: Failed to initialize pipeline: {str(e)}")
    raise

class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend at the root endpoint."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/query/text")
async def text_query(request: QueryRequest):
    """Handle text-based queries with integrated TTS and MongoDB storage."""
    query = request.query.strip()
    user_id = request.user_id.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")
    
    try:
        results = pipeline.search(
            query,
            top_k=config['top_k'],
            faiss_weight=config['faiss_weight'],
            bm25_weight=config['bm25_weight']
        )
        chunk_ids = [chunk_id for chunk_id, score in results]
        if not chunk_ids:
            pipeline.log_search(query, [], "No relevant chunks found", config['csv_log'])
            await user_chats_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {
                        "chats": {
                            "query": query,
                            "answer": "No relevant chunks found",
                            "transcription": None,
                            "timestamp": datetime.utcnow().isoformat(),
                            "type": "text"
                        }
                    }
                },
                upsert=True
            )
            return {"answer": "No relevant chunks found", "tts_audio": ""}
        
        answer = await pipeline.generate_answer(query, chunk_ids, user_id)
        pipeline.log_search(query, chunk_ids, answer, config['csv_log'])
        
        await user_chats_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "chats": {
                        "query": query,
                        "answer": answer,
                        "transcription": None,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "text"
                    }
                }
            },
            upsert=True
        )
        
        tts_audio_base64 = ""
        if not answer.startswith("Error"):
            try:
                response = await asyncio.to_thread(
                    pipeline.llm_client.audio.speech.create,
                    model="tts-1",
                    voice="alloy",
                    input=answer,
                    response_format="mp3"
                )
                audio_data = response.content
                tts_audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                logger.info("TTS-GENERATE: Generated TTS audio for answer")
            except Exception as e:
                logger.error(f"TTS-ERROR: Failed to generate TTS: {str(e)}")
        
        return {"answer": answer, "tts_audio": tts_audio_base64}
    except Exception as e:
        pipeline.log_search(query, [], f"Error: {str(e)}", config['csv_log'])
        await user_chats_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "chats": {
                        "query": query,
                        "answer": f"Error: {str(e)}",
                        "transcription": None,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "text"
                    }
                }
            },
            upsert=True
        )
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/voice")
async def voice_query(file: UploadFile = File(...), user_id: str = Form(...)):
    """Handle voice-based queries with integrated TTS and MongoDB storage."""
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")
    
    try:
        audio_data = await file.read()
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"
        
        transcription = await asyncio.to_thread(
            pipeline.llm_client.audio.transcriptions.create,
            model="whisper-1",
            file=audio_file
        )
        query = transcription.text.strip()
        logger.info(f"WHISPER-TRANSCRIBE: Transcribed audio to: {query}")
        
        if not query:
            raise HTTPException(status_code=400, detail="Transcribed query is empty")
        
        results = pipeline.search(
            query,
            top_k=config['top_k'],
            faiss_weight=config['faiss_weight'],
            bm25_weight=config['bm25_weight']
        )
        chunk_ids = [chunk_id for chunk_id, score in results]
        if not chunk_ids:
            pipeline.log_search(query, [], "No relevant chunks found", config['csv_log'])
            await user_chats_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {
                        "chats": {
                            "query": query,
                            "answer": "No relevant chunks found",
                            "transcription": query,
                            "timestamp": datetime.utcnow().isoformat(),
                            "type": "voice"
                        }
                    }
                },
                upsert=True
            )
            return {"answer": "No relevant chunks found", "transcription": query, "tts_audio": ""}
        
        answer = await pipeline.generate_answer(query, chunk_ids, user_id)
        pipeline.log_search(query, chunk_ids, answer, config['csv_log'])
        
        await user_chats_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "chats": {
                        "query": query,
                        "answer": answer,
                        "transcription": query,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "voice"
                    }
                }
            },
            upsert=True
        )
        
        tts_audio_base64 = ""
        if not answer.startswith("Error"):
            try:
                response = await asyncio.to_thread(
                    pipeline.llm_client.audio.speech.create,
                    model="tts-1",
                    voice="alloy",
                    input=answer,
                    response_format="mp3"
                )
                audio_data = response.content
                tts_audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                logger.info("TTS-GENERATE: Generated TTS audio for answer")
            except Exception as e:
                logger.error(f"TTS-ERROR: Failed to generate TTS: {str(e)}")
        
        return {"answer": answer, "transcription": query, "tts_audio": tts_audio_base64}
    except Exception as e:
        pipeline.log_search("", [], f"Error: {str(e)}", config['csv_log'])
        await user_chats_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "chats": {
                        "query": "",
                        "answer": f"Error: {str(e)}",
                        "transcription": None,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "voice"
                    }
                }
            },
            upsert=True
        )
        raise HTTPException(status_code=500, detail=f"Error processing voice query: {str(e)}")

def run_cli():
    """Run the original CLI interface."""
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot with FastAPI or CLI")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of FastAPI")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8005)