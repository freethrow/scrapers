import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.model = None
        self.client = None
        self.db = None
        self.collection = None
        self.batch_size = 10

    def connect_db(self):
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
            )
            # Test connection
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def load_model(self):
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer("djovak/embedic-large")
            logger.info("Successfully loaded embedding model")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_articles_without_embeddings(self):
        """Get articles that need embeddings."""
        try:
            # Query for documents where embedding is null or doesn't exist
            query = {"$or": [{"embedding": None}, {"embedding": {"$exists": False}}]}

            # Get total count
            total_count = self.collection.count_documents(query)
            logger.info(f"Found {total_count} articles without embeddings")

            return self.collection.find(query)

        except Exception as e:
            logger.error(f"Error querying articles: {str(e)}")
            raise

    def process_batch(self, articles):
        """Process a batch of articles and update their embeddings."""
        try:
            for article in articles:
                try:
                    # Combine title and content for embedding
                    text_for_embedding = f"{article['title']} {article['content']}"

                    # Generate embedding
                    embedding = self.model.encode(text_for_embedding)

                    # Update the document in MongoDB
                    self.collection.update_one(
                        {"_id": article["_id"]},
                        {"$set": {"embedding": embedding.tolist()}},
                    )

                    logger.info(f"Generated embedding for article: {article['title']}")

                except Exception as e:
                    logger.error(
                        f"Error processing article {article.get('_id')}: {str(e)}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    def process_all_articles(self):
        """Process all articles that need embeddings in batches."""
        try:
            # Get cursor for articles without embeddings
            articles_cursor = self.get_articles_without_embeddings()

            # Process in batches
            current_batch = []
            processed_count = 0

            for article in articles_cursor:
                current_batch.append(article)

                if len(current_batch) >= self.batch_size:
                    self.process_batch(current_batch)
                    processed_count += len(current_batch)
                    logger.info(f"Processed {processed_count} articles")
                    current_batch = []
                    time.sleep(1)  # Brief pause between batches

            # Process any remaining articles
            if current_batch:
                self.process_batch(current_batch)
                processed_count += len(current_batch)
                logger.info(f"Processed {processed_count} articles")

        except Exception as e:
            logger.error(f"Error in process_all_articles: {str(e)}")
            raise
        finally:
            if self.client:
                self.client.close()
                logger.info("Closed MongoDB connection")


def main():
    """Run the embedding generation process."""
    try:
        # Verify environment variables
        required_vars = ["MONGODB_URI", "DB_NAME", "COLLECTION_NAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize the embedding generator
        generator = EmbeddingGenerator()

        # Connect to database
        generator.connect_db()

        # Load the model
        generator.load_model()

        # Process all articles
        generator.process_all_articles()

        logger.info("Embedding generation completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
