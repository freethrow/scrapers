# ekapija.py

from datetime import datetime
import os
from dotenv import load_dotenv

import logging
import random

import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DropItem
from scrapy import signals
import pytz

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class MongoDBPipeline:
    """Pipeline for storing scraped items in MongoDB."""

    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_ENG")
        self.client = None
        self.db = None
        self.collection = None

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def open_spider(self, spider):
        """Initialize MongoDB connection when spider opens."""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
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

    def close_spider(self, spider):
        """Clean up MongoDB connection when spider closes."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    def process_item(self, item, spider):
        """Process and store items in MongoDB."""
        try:
            # Add timestamp
            item["scraped_at"] = datetime.utcnow()

            # Insert into MongoDB
            self.collection.insert_one(dict(item))
            logger.info(f"Successfully saved article: {item['title']}")
            return item

        except DuplicateKeyError:
            logger.info(f"Duplicate article found: {item['title']}")
            raise DropItem(f"Duplicate article found: {item['title']}")

        except Exception as e:
            logger.error(f"Error saving article to MongoDB: {str(e)}")
            raise


class EkapijaSpider(CrawlSpider):
    name = "ekapijaspider"
    allowed_domains = ["ekapija.com"]
    start_urls = [
        "https://www.ekapija.com/en/news/agro",
        "https://www.ekapija.com/en/news/energija",
        "https://www.ekapija.com/en/news/industrija",
        "https://www.ekapija.com/en/news/gradjevina",
        "https://www.ekapija.com/en/news/saobracaj",
        "https://www.ekapija.com/en/news/it-telekomunikacije",
    ]

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 20,
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
        # Rule for article pages
        Rule(
            LinkExtractor(
                allow=r"/en/news/\d+/.*",
                restrict_xpaths="//div[contains(@class, 'box_R') or contains(@class, 'box_BL') or contains(@class, 'box_LD') or contains(@class, 'box_RD')]//a",
            ),
            callback="parse_article",
            follow=False,
        ),
        # Rule for pagination - page 2 for each section
        Rule(
            LinkExtractor(
                allow=[
                    r"/en/news/agro\?page=2$",
                    r"/en/news/energija\?page=2$",
                    r"/en/news/industrija\?page=2$",
                    r"/en/news/gradjevina\?page=2$",
                    r"/en/news/saobracaj\?page=2$",
                    r"/en/news/it-telekomunikacije\?page=2$",
                ],
                restrict_xpaths="//div[contains(@class, 'contBox')]//ul[contains(@class, 'pagination')]",
            ),
            follow=True,
        ),
    ]

    def parse_article(self, response):
        """Extract article data from the response."""
        try:
            # Extract title from h1
            title = response.css("h1::text").get()
            if title:
                title = title.strip()

            # Extract date from the source div
            date_div = response.css("div.sourceN")
            date_str = date_div.xpath(".//span[contains(text(), '.')]//text()").get()
            time_str = date_div.xpath(".//span[last()]/text()").get()

            if date_str and time_str:
                try:
                    # Clean date string (remove day name if present)
                    date_str = date_str.strip()
                    if "," in date_str:
                        date_str = date_str.split(",")[1].strip()

                    # Now date_str looks like "25.12.2024."
                    # Remove trailing dot if present
                    date_str = date_str.rstrip(".")

                    time_str = time_str.strip()

                    # Split date parts and construct in correct order
                    day, month, year = date_str.split(".")
                    formatted_date = (
                        f"{year}-{month}-{day}"  # Convert to YYYY-MM-DD format
                    )

                    # Combine date and time
                    date_time_str = f"{formatted_date} {time_str}"
                    naive_date = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")

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

            # Extract content from div.txtBoxN
            content_blocks = []
            for block in response.css("div.txtBoxN"):
                # Get all text nodes
                text_parts = []
                text_parts.extend(
                    block.xpath(
                        ".//text()[not(parent::script)][not(parent::style)][normalize-space()]"
                    ).getall()
                )
                text_parts.extend(block.css("p::text").getall())

                if text_parts:
                    content_blocks.extend(text_parts)

            # Clean and join content
            content = " ".join(
                [
                    part.strip()
                    for part in content_blocks
                    if part.strip()
                    and not part.strip().startswith("Comments")
                    and not part.strip().startswith("Companies:")
                    and not part.strip().startswith("Tags:")
                ]
            )

            # Create item
            item = {
                "title": title,
                "date": date,
                "content": content,
                "url": response.url,
                "source": "ekapija.com",
            }

            logger.info(f"Successfully parsed article: {title}")
            return item

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {str(e)}")
            return None

    def closed(self, reason):
        """Log when spider is closed."""
        logger.info(f"Spider closed: {reason}")


class RandomUserAgentMiddleware:
    """Middleware to rotate User-Agents for each request."""

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
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


def main():
    """Run the spider."""
    try:
        # Verify environment variables
        required_vars = ["MONGODB_URI", "DB_NAME", "COLLECTION_ENG"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        process = CrawlerProcess(
            {
                "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "LOG_LEVEL": "INFO",
            }
        )

        logger.info("Starting spider...")
        process.crawl(EkapijaSpider)
        process.start()

    except Exception as e:
        logger.error(f"Error running spider: {str(e)}")
        raise


if __name__ == "__main__":
    main()
