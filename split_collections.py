import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class CollectionSplitter:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.source_collection = "ai_articles"
        self.serbian_collection = "ai_articles_serbian"
        self.english_collection = "ai_articles_english"
        self.client = None
        self.db = None

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

    def create_new_collections(self):
        """Create new collections if they don't exist."""
        try:
            if self.english_collection not in self.db.list_collection_names():
                self.db.create_collection(self.english_collection)
                logger.info(f"Created new collection: {self.english_collection}")

            logger.info("Collections setup completed")

        except Exception as e:
            logger.error(f"Error creating collections: {str(e)}")
            raise

    def move_ekapija_articles(self):
        """Move Ekapija articles to English collection."""
        try:
            # Find all Ekapija articles
            query = {"source": "ekapija.com"}
            ekapija_articles = self.db[self.source_collection].find(query)

            # Count articles to move
            total_articles = self.db[self.source_collection].count_documents(query)
            logger.info(f"Found {total_articles} Ekapija articles to move")

            # Move articles in batches
            batch_size = 100
            processed = 0

            while processed < total_articles:
                batch = self.db[self.source_collection].find(query).limit(batch_size)

                # Insert into new collection
                if batch:
                    articles = list(batch)
                    self.db[self.english_collection].insert_many(articles)

                    # Remove from original collection
                    ids = [article["_id"] for article in articles]
                    self.db[self.source_collection].delete_many({"_id": {"$in": ids}})

                    processed += len(articles)
                    logger.info(f"Moved {processed}/{total_articles} articles")

            logger.info("Completed moving Ekapija articles")

        except Exception as e:
            logger.error(f"Error moving articles: {str(e)}")
            raise

    def rename_serbian_collection(self):
        """Rename the original collection to ai_articles_serbian."""
        try:
            self.db[self.source_collection].rename(self.serbian_collection)
            logger.info(f"Renamed collection to {self.serbian_collection}")

        except Exception as e:
            logger.error(f"Error renaming collection: {str(e)}")
            raise

    def process(self):
        """Execute the full collection split process."""
        try:
            # Connect to database
            self.connect_db()

            # Create new collections
            self.create_new_collections()

            # Move Ekapija articles
            self.move_ekapija_articles()

            # Rename original collection
            self.rename_serbian_collection()

            logger.info("Collection split completed successfully")

        except Exception as e:
            logger.error(f"Error during collection split: {str(e)}")
            raise
        finally:
            if self.client:
                self.client.close()
                logger.info("Closed MongoDB connection")


def main():
    """Run the collection split process."""
    try:
        # Verify environment variables
        required_vars = ["MONGODB_URI", "DB_NAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize and run the splitter
        splitter = CollectionSplitter()
        splitter.process()

    except Exception as e:
        logger.error(f"Error during collection split: {str(e)}")
        raise


if __name__ == "__main__":
    main()
