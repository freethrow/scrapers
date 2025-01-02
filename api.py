# api.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
import time
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Initialize MongoDB connection
class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        """Create a MongoDB connection with retry logic."""
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                self.client = MongoClient(
                    os.getenv("MONGODB_URI"),
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000,
                    maxPoolSize=10,
                    minPoolSize=5,
                    retryWrites=True,
                    retryReads=True,
                    directConnection=False,
                )

                # Test connection
                self.client.admin.command("ping")

                self.db = self.client[os.getenv("DB_NAME")]
                self.collection = self.db[os.getenv("COLLECTION_SRB")]

                logger.info("Successfully connected to MongoDB")
                return

            except Exception as e:
                logger.error(
                    f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    raise

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")


# Global instances
mongodb = MongoDB()
model = None
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and connect to MongoDB
    try:
        global model
        model = SentenceTransformer("djovak/embedic-large")
        logger.info("Successfully loaded embedding model")

        mongodb.connect()

        yield
    finally:
        # Shutdown: Clean up resources
        mongodb.close()


# Initialize FastAPI app
app = FastAPI(title="Article Vector Search API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5


class SearchResult(BaseModel):
    title: str
    content: str
    date: Optional[datetime]
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float


class AIQuery(BaseModel):
    topic: str
    additional_context: Optional[str] = ""
    language: Optional[str] = "English"
    top_k: Optional[int] = 40


class AIResponse(BaseModel):
    topic: str
    report: str
    search_time: float
    processing_time: float


@app.post("/search", response_model=SearchResponse)
async def search_articles(search_query: SearchQuery):
    """
    Perform semantic search on articles.
    """
    try:
        start_time = datetime.now()

        # Generate embedding for query
        query_embedding = model.encode(search_query.query).tolist()

        # Prepare aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": search_query.limit,
                }
            },
            {
                "$project": {
                    "title": 1,
                    "content": 1,
                    "date": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        # Execute search
        results = list(mongodb.collection.aggregate(pipeline))

        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds()

        # Prepare response
        search_response = SearchResponse(
            results=[SearchResult(**result) for result in results],
            query=search_query.query,
            total_results=len(results),
            search_time=search_time,
        )

        return search_response

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai", response_model=AIResponse)
async def generate_ai_report(query: AIQuery):
    """
    Generate an AI report based on semantic search results using Claude.
    """
    try:
        start_time = datetime.now()

        # Generate embedding for query
        query_embedding = model.encode(query.topic).tolist()

        # Prepare aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": query.top_k,
                }
            },
            {
                "$project": {
                    "title": 1,
                    "content": 1,
                    "date": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        # Execute search
        results = list(mongodb.collection.aggregate(pipeline))
        search_time = (datetime.now() - start_time).total_seconds()

        # Prepare articles text for Claude
        articles_text = "\n\n".join(
            [
                f"Article {i+1}:\nDate: {doc.get('date', 'N/A')}\nTitle: {doc['title']}\nContent: {doc['content'][:500]}..."
                for i, doc in enumerate(results)
            ]
        )

        # Create prompt for report generation
        prompt = f"""You are an expert journalist and news analyst. Based on the following {query.top_k} most relevant Serbian news articles about "{query.topic}", 
create a concise, well-structured news report in {query.language}. The report should:
- Be around 2000-2500 words
- Start with a clear headline
- Include key facts, dates, and relevant context
- Maintain journalistic style
- Focus on the most newsworthy aspects
- Include relevant news sources
- Include a brief conclusion or outlook
- Mark specific topics with bold or italics - topics that could be further analyzed

{f'Additional context to consider: {query.additional_context}' if query.additional_context else ''}

Here are the articles:
{articles_text}

Please write the report in a professional journalistic style."""

        # Get response from Claude
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        report = response.content[0].text.strip()
        total_time = (datetime.now() - start_time).total_seconds()

        return AIResponse(
            topic=query.topic,
            report=report,
            search_time=search_time,
            processing_time=total_time,
        )

    except Exception as e:
        logger.error(f"AI Report generation error: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
