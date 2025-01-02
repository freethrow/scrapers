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
import psutil
import humanize

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage of the script."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = datetime.now()
        self.start_memory = self.get_memory_usage()
        self.peak_memory = self.start_memory
        self.measurements = []

    def get_memory_usage(self):
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss

    def measure(self, checkpoint_name):
        """Record memory usage at a specific checkpoint."""
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)

        measurement = {
            "checkpoint": checkpoint_name,
            "timestamp": datetime.now(),
            "memory_usage": current_memory,
            "memory_increase": current_memory - self.start_memory,
        }
        self.measurements.append(measurement)

        # Log the measurement
        logger.info(
            f"Memory at {checkpoint_name}: "
            f"{humanize.naturalsize(current_memory)} "
            f"(Δ: {humanize.naturalsize(measurement['memory_increase'])})"
        )

    def summary(self):
        """Generate a summary of memory usage."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        summary = (
            f"\nMemory Usage Summary:\n"
            f"Duration: {duration}\n"
            f"Initial Memory: {humanize.naturalsize(self.start_memory)}\n"
            f"Peak Memory: {humanize.naturalsize(self.peak_memory)}\n"
            f"Peak Memory Increase: {humanize.naturalsize(self.peak_memory - self.start_memory)}\n"
            f"\nCheckpoint Details:"
        )

        for m in self.measurements:
            summary += f"\n{m['checkpoint']}:\n"
            summary += f"  Usage: {humanize.naturalsize(m['memory_usage'])}\n"
            summary += f"  Increase: {humanize.naturalsize(m['memory_increase'])}\n"

        logger.info(summary)


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
    """Extract content from a div with class 'entry-content' using BeautifulSoup with UTF-8 encoding."""
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
    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_SRB")
        self.client = None
        self.db = None
        self.collection = None

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
        # Initialize memory monitor
        memory_monitor = MemoryMonitor()
        memory_monitor.measure("Script Start")

        # Verify environment variables
        required_vars = ["MONGODB_URI", "DB_NAME", "COLLECTION_SRB"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize MongoDB
        mongodb = MongoDBHandler()
        mongodb.connect()
        memory_monitor.measure("MongoDB Connection")

        logger.info("Starting scraper...")

        # Process pages one by one
        page = 1
        max_pages = 10  # You can adjust this limit

        while page <= max_pages:
            try:
                logger.info(f"Starting to process page {page}")
                articles_found = process_page(page, mongodb)
                memory_monitor.measure(f"After Page {page}")

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
        memory_monitor.measure("After MongoDB Close")

        # Generate memory usage summary
        memory_monitor.summary()

        logger.info("Scraping completed successfully")

    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise


if __name__ == "__main__":
    main()
