import logging
from datetime import datetime
import random
import os
from functools import wraps
import time
from dotenv import load_dotenv
import argparse

import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DropItem
from scrapy import signals
from sentence_transformers import SentenceTransformer
import pytz

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Retry decorator with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep_time = backoff_in_seconds * 2**x + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    x += 1

        return wrapper

    return decorator


class RandomUserAgentMiddleware:
    """Middleware to rotate User-Agents for each request."""

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]

    def __init__(self):
        self.user_agents = self.user_agents

    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def process_request(self, request, spider):
        user_agent = random.choice(self.user_agents)
        request.headers["User-Agent"] = user_agent
        return None

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class MongoDBPipeline:
    """Pipeline for storing scraped items in MongoDB."""

    def __init__(self, enable_embeddings=False):
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.client = None
        self.db = None
        self.collection = None
        self.enable_embeddings = enable_embeddings
        self.model = None

        # Only initialize the model if embeddings are enabled
        if self.enable_embeddings:
            self.model = SentenceTransformer("djovak/embedic-large")
            logger.info("Initialized embedding model")

    @classmethod
    def from_crawler(cls, crawler):
        # Get the embeddings setting from crawler settings
        enable_embeddings = crawler.settings.get("ENABLE_EMBEDDINGS", False)
        return cls(enable_embeddings=enable_embeddings)

    def open_spider(self, spider):
        """Initialize MongoDB connection when spider opens."""
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
            # Test the connection
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

            # Create index on URL to ensure uniqueness
            self.collection.create_index([("url", pymongo.ASCENDING)], unique=True)

        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def close_spider(self, spider):
        """Clean up MongoDB connection when spider closes."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    @retry_with_backoff(retries=3)
    def process_item(self, item, spider):
        """Process and store items in MongoDB with error handling and retries."""
        try:
            # Add timestamp
            item["scraped_at"] = datetime.utcnow()

            # Generate embedding only if enabled
            if self.enable_embeddings and self.model:
                try:
                    text_for_embedding = f"{item['title']} {item['content']}"
                    embedding = self.model.encode(text_for_embedding)
                    item["embedding"] = embedding.tolist()
                    logger.info(f"Generated embedding for article: {item['title']}")
                except Exception as e:
                    logger.error(f"Error generating embedding: {str(e)}")
                    item["embedding"] = None
            else:
                item["embedding"] = None

            # Insert the item
            self.collection.insert_one(dict(item))
            logger.info(f"Successfully saved article: {item['title']}")
            return item

        except DuplicateKeyError:
            logger.info(f"Duplicate article found: {item['title']}")
            raise DropItem(f"Duplicate article found: {item['title']}")

        except Exception as e:
            logger.error(f"Error saving article to MongoDB: {str(e)}")
            raise


class DanasSpider(CrawlSpider):
    name = "danasspider"
    allowed_domains = ["danas.rs"]
    start_urls = [
        "https://www.danas.rs/rubrika/vesti/politika/",
        "https://www.danas.rs/rubrika/vesti/drustvo/",
    ]

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 2,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 8,
        "COOKIES_ENABLED": False,
        "DOWNLOAD_TIMEOUT": 20,
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
            __name__ + ".RandomUserAgentMiddleware": 400,
        },
        "ITEM_PIPELINES": {
            __name__ + ".MongoDBPipeline": 300,
        },
    }

    rules = [
        Rule(
            LinkExtractor(
                allow=r"https://www.danas.rs/vesti/politika/*",
            ),
            callback="parse_article",
            follow=False,
        ),
        Rule(
            LinkExtractor(
                allow=r"https://www.danas.rs/vesti/drustvo/*",
            ),
            callback="parse_article",
            follow=False,
        ),
        Rule(
            LinkExtractor(
                allow=r"https://www.danas.rs/vesti/ekonomija/*",
            ),
            callback="parse_article",
            follow=False,
        ),
        Rule(
            LinkExtractor(
                allow=r"https://www.danas.rs/rubrika/vesti/politika/page/(?:[1-9]|[1-9][0-9])/\Z",
            ),
            follow=True,
        ),
        Rule(
            LinkExtractor(
                allow=r"https://www.danas.rs/rubrika/vesti/drustvo/page/(?:[1-9]|[1-9][0-9])/\Z",
            ),
            follow=True,
        ),
        Rule(
            LinkExtractor(
                allow=r"https://www.danas.rs/rubrika/vesti/ekonomija/page/(?:[1-9]|[1-9][0-9])/\Z",
            ),
            follow=True,
        ),
    ]

    def parse_article(self, response):
        """Extract article data from the response."""
        try:
            title = (
                response.css("h1.post-title::text")
                .get()
                .replace('"', "")
                .replace("'", "")
                .replace('"', "")
                .replace('"', "")
            )

            # Extract date parts
            date_str = response.css("span.special-date::text").get()
            time_str = response.css("span.special-time::text").get()

            if date_str and time_str:
                try:
                    date_str = date_str.strip().lower()
                    time_str = time_str.strip()

                    # Handle "danas" case
                    if "danas" in date_str:
                        # Use today's date
                        today = datetime.now()
                        date_str = today.strftime("%d.%m.%Y.")

                    # Combine date and time strings and parse
                    date_time_str = f"{date_str} {time_str}"
                    naive_date = datetime.strptime(date_time_str, "%d.%m.%Y. %H:%M")

                    # Add timezone (Serbian time)
                    serbian_tz = pytz.timezone("Europe/Belgrade")
                    local_date = serbian_tz.localize(naive_date)

                    # Convert to UTC for MongoDB
                    date = local_date.astimezone(pytz.UTC)
                    logger.info(f"Successfully parsed date: {date_time_str} to {date}")
                except Exception as e:
                    logger.error(f"Error parsing date '{date_time_str}': {e}")
                    date = None
            else:
                logger.warning(f"No date found for {response.url}")
                date = None

            content = "".join(
                [
                    p.strip()
                    .replace('"', "")
                    .replace("'", "")
                    .replace('"', "")
                    .replace('"', "")
                    for p in response.css(".content.post-content p::text").getall()
                    if p.strip()
                ]
            )

            item = {
                "title": title,
                "date": date,
                "content": content,
                "url": response.url,
                "source": "danas.rs",
            }

            logger.info(f"Successfully parsed article: {title}")
            return item

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {str(e)}")
            return None

    def closed(self, reason):
        """Log when spider is closed."""
        logger.info(f"Spider closed: {reason}")


def main():
    """Run the spider."""
    try:
        # Add argument parsing
        parser = argparse.ArgumentParser(description="Run the Danas spider")
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

        process = CrawlerProcess(
            {
                "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "LOG_LEVEL": "INFO",
                "ENABLE_EMBEDDINGS": args.embeddings,  # Pass the embeddings flag to the settings
            }
        )

        logger.info(
            f"Starting spider with embeddings {'enabled' if args.embeddings else 'disabled'}..."
        )
        process.crawl(DanasSpider)
        process.start()

    except Exception as e:
        logger.error(f"Error running spider: {str(e)}")
        raise


if __name__ == "__main__":
    main()
