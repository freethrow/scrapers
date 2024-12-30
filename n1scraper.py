import requests
import re
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
from dateutil.parser import parse as dateutil_parse
import pytz
import os
from dotenv import load_dotenv
import logging
import pymongo
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import argparse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Model will be initialized only if embeddings are enabled
model = None


def get_random_headers():
    """Get random browser headers."""
    headers = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
    ]
    return random.choice(headers)


def process_page(page_num, mongodb_handler):
    """Process a single page of articles."""
    logger.info(f"Processing page {page_num}")
    time.sleep(random.uniform(3, 5))  # Random delay between pages

    json_url = (
        f"https://n1info.rs/wp-json/wp/v2/uc-all-posts?page={page_num}&per_page=50"
    )
    response = requests.get(json_url, headers=get_random_headers())
    data = response.json()

    articles_found = False

    for item in data["data"]:
        if any(section in item["link"] for section in ["vesti", "biznis", "region"]):
            articles_found = True
            url = item["link"]

            # Check if article already exists
            if mongodb_handler.article_exists(url):
                logger.info(f"Article already exists: {url}")
                continue

            logger.info(f"Processing new article: {url}")
            time.sleep(random.uniform(2, 4))  # Random delay between articles

            article = extract_entry_content(url)
            if isinstance(article, dict) and "error" not in article:
                mongodb_handler.save_article(article)

    return articles_found


def clean_text(text):
    """Clean the text by removing extra whitespace and specific unwanted text."""
    # Remove "Podeli:" and similar text
    text = re.sub(r"Podeli\s*:", "", text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    return text.strip()


def extract_entry_content(url):
    """
    Extract content from a div with class 'entry-content' using BeautifulSoup with UTF-8 encoding.
    """
    try:
        # Fetch the webpage with explicit encoding
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Charset": "utf-8",
        }
        response = requests.get(url, headers=headers)
        response.encoding = "utf-8"

        # Create BeautifulSoup object
        soup = BeautifulSoup(response.text, "html.parser", from_encoding="utf-8")

        # Find the entry-content div
        entry_content = soup.find("div", class_="entry-content")
        title = soup.find("h1", class_="entry-title")
        date_div = soup.find("span", class_="post-time")
        date = date_div.find("span") if date_div else None

        if entry_content:
            # Remove share buttons or similar elements if they exist
            share_elements = entry_content.find_all(
                class_=lambda x: x and ("share" in x.lower() or "social" in x.lower())
            )
            for element in share_elements:
                element.decompose()

            # Get and clean the text content
            text_content = clean_text(entry_content.get_text(separator="\n"))

            # Clean paragraphs individually
            paragraphs = [clean_text(p.get_text()) for p in entry_content.find_all("p")]
            # Remove empty paragraphs
            paragraphs = [p for p in paragraphs if p]

            # Parse date string to MongoDB compatible format
            date_str = date.get_text().strip() if date else None
            if date_str:
                try:
                    parsed_date = dateutil_parse(date_str)
                    if parsed_date.tzinfo is None:
                        parsed_date = pytz.timezone("Europe/Belgrade").localize(
                            parsed_date
                        )
                    mongodb_date = parsed_date.astimezone(pytz.UTC)
                except Exception as e:
                    logger.error(f"Error parsing date {date_str}: {e}")
                    mongodb_date = None
            else:
                mongodb_date = None

            return {
                "content": text_content,
                "title": title.get_text().strip(),
                "date": mongodb_date,
                "url": url,
                "source": "n1info",
            }
        else:
            return {"error": "No div with class entry-content found"}

    except requests.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except UnicodeError as e:
        return {"error": f"Encoding error: {str(e)}"}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}


class MongoDBHandler:
    def __init__(self, enable_embeddings=False):
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.client = None
        self.db = None
        self.collection = None
        self.enable_embeddings = enable_embeddings

        # Only initialize the model if embeddings are enabled
        global model
        if self.enable_embeddings and model is None:
            model = SentenceTransformer("djovak/embedic-large")
            logger.info("Initialized embedding model")

    def connect(self):
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
                maxPoolSize=10,
                minPoolSize=5,
                retryWrites=True,
                retryReads=True,
            )
            # Test connection
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

            # Create index on URL
            self.collection.create_index([("url", pymongo.ASCENDING)], unique=True)

        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def article_exists(self, url):
        try:
            return bool(self.collection.find_one({"url": url}))
        except Exception as e:
            logger.error(f"Error checking article existence: {str(e)}")
            return False

    def save_article(self, article):
        try:
            # Add timestamp
            article["scraped_at"] = datetime.utcnow()

            # Generate embedding only if enabled and model is loaded
            if self.enable_embeddings and model is not None:
                try:
                    text_for_embedding = f"{article['title']} {article['content']}"
                    embedding = model.encode(text_for_embedding)
                    article["embedding"] = embedding.tolist()
                    logger.info(f"Generated embedding for article: {article['title']}")
                except Exception as e:
                    logger.error(f"Error generating embedding: {str(e)}")
                    article["embedding"] = None
            else:
                article["embedding"] = None

            # Save to MongoDB
            self.collection.insert_one(article)
            logger.info(f"Successfully saved article: {article['title']}")
            return True

        except pymongo.errors.DuplicateKeyError:
            logger.info(f"Duplicate article found: {article['title']}")
            return False
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}")
            raise

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")


def main():
    try:
        # Add argument parsing
        parser = argparse.ArgumentParser(description="Run the N1 scraper")
        parser.add_argument(
            "--embeddings",
            action="store_true",
            default=False,
            help="Enable embeddings generation (default: False)",
        )
        args = parser.parse_args()

        # Verify environment variables
        required_vars = ["MONGODB_URI", "DB_NAME", "COLLECTION_NAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize MongoDB with embeddings flag
        mongodb = MongoDBHandler(enable_embeddings=args.embeddings)
        mongodb.connect()

        logger.info(
            f"Starting scraper with embeddings {'enabled' if args.embeddings else 'disabled'}..."
        )

        # Process pages one by one
        page = 1
        max_pages = 50  # You can adjust this limit

        while page <= max_pages:
            try:
                logger.info(f"Starting to process page {page}")
                articles_found = process_page(page, mongodb)

                # If no articles found on the page, we might have reached the end
                if not articles_found:
                    logger.info(f"No articles found on page {page}, stopping...")
                    break

                page += 1

            except Exception as e:
                logger.error(f"Error processing page {page}, stopping: {str(e)}")
                break

        # Cleanup
        mongodb.close()
        logger.info("Scraping completed successfully")

    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise


if __name__ == "__main__":
    main()
