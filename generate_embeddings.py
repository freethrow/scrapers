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
        self.serbian_collection = os.getenv("COLLECTION_NAME")
        self.english_collection = os.getenv("COLLECTION_ENG")
        self.serbian_model = None
        self.english_model = None
        self.client = None
        self.db = None
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

        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def load_models(self):
        """Initialize both embedding models."""
        try:
            # Load Serbian model
            self.serbian_model = SentenceTransformer("djovak/embedic-large")
            logger.info("Successfully loaded Serbian embedding model")

            # Load English model
            self.english_model = SentenceTransformer(
                "ibm-granite/granite-embedding-278m-multilingual"
            )
            logger.info("Successfully loaded English embedding model")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def get_articles_without_embeddings(self, collection_name, is_english=False):
        """Get articles that need embeddings from specified collection."""
        try:
            collection = self.db[collection_name]

            # Query for documents where embedding is null or doesn't exist
            query = {"$or": [{"embedding": None}, {"embedding": {"$exists": False}}]}
            if is_english:
                query["source"] = "ekapija.com"

            # Get total count
            total_count = collection.count_documents(query)
            logger.info(
                f"Found {total_count} articles without embeddings in {collection_name}"
            )

            return collection.find(query)

        except Exception as e:
            logger.error(f"Error querying articles: {str(e)}")
            raise

    def process_batch(self, articles, collection_name, is_english=False):
        """Process a batch of articles and update their embeddings."""
        try:
            collection = self.db[collection_name]
            model = self.english_model if is_english else self.serbian_model

            for article in articles:
                try:
                    # Combine title and content for embedding
                    text_for_embedding = f"{article['title']} {article['content']}"

                    # Generate embedding
                    embedding = model.encode(text_for_embedding)

                    # Update the document in MongoDB
                    collection.update_one(
                        {"_id": article["_id"]},
                        {"$set": {"embedding": embedding.tolist()}},
                    )

                    logger.info(
                        f"Generated {'English' if is_english else 'Serbian'} embedding for article: {article['title']}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing article {article.get('_id')}: {str(e)}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    def process_collection(self, collection_name, is_english=False):
        """Process all articles in a collection that need embeddings."""
        try:
            # Get cursor for articles without embeddings
            articles_cursor = self.get_articles_without_embeddings(
                collection_name, is_english
            )

            # Process in batches
            current_batch = []
            processed_count = 0

            for article in articles_cursor:
                current_batch.append(article)

                if len(current_batch) >= self.batch_size:
                    self.process_batch(current_batch, collection_name, is_english)
                    processed_count += len(current_batch)
                    logger.info(
                        f"Processed {processed_count} articles in {collection_name}"
                    )
                    current_batch = []
                    time.sleep(1)  # Brief pause between batches

            # Process any remaining articles
            if current_batch:
                self.process_batch(current_batch, collection_name, is_english)
                processed_count += len(current_batch)
                logger.info(
                    f"Processed {processed_count} articles in {collection_name}"
                )

        except Exception as e:
            logger.error(f"Error in process_collection: {str(e)}")
            raise

    def process_all_articles(self):
        """Process all articles that need embeddings in both collections."""
        try:
            # First process Serbian articles
            logger.info("Starting Serbian articles processing...")
            self.process_collection(self.serbian_collection, is_english=False)
            logger.info("Completed Serbian articles processing")

            # Then process English articles
            logger.info("Starting English articles processing...")
            self.process_collection(self.english_collection, is_english=True)
            logger.info("Completed English articles processing")

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
        required_vars = ["MONGODB_URI", "DB_NAME", "COLLECTION_NAME", "COLLECTION_ENG"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize the embedding generator
        generator = EmbeddingGenerator()

        # Connect to database
        generator.connect_db()

        # Load the models
        generator.load_models()

        # Process all articles
        generator.process_all_articles()

        logger.info("Embedding generation completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
