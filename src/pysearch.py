#!/usr/bin/env python3
"""
Instant answer and topic search tool using DuckDuckGo API.

This script provides a command-line tool to fetch instant answers, definitions,
and related topics for queries using the DuckDuckGo Instant Answer API.
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
import logging

# ANSI color codes
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"


logger = logging.getLogger(__name__)


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that adds ANSI color codes to log messages.

    This formatter applies color coding based on log levels for better
    readability in terminal output.
    """

    COLORS = {
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Args:
            record (logging.LogRecord):
                The log record to format.

        Returns:
            str:
                The formatted log message with ANSI color codes.

        """
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message


def print_colored(text: str, color: str) -> None:
    """Print text with ANSI color codes."""
    print(f"{color}{text}{RESET}")


def fetch_answer(query: str) -> dict:
    """Fetch search results for a query from DuckDuckGo API."""
    # Fetch from DuckDuckGo search API
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
    logger.debug(f"Making API request to: {url}")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            logger.debug(f"Received response with {len(data)} top-level keys")
    except urllib.error.URLError as e:
        logger.error(f"Error fetching data: {e}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return {}
    return data


def main():
    """Main entry point for the instant answer search tool."""
    parser = argparse.ArgumentParser(
        description="Fetch instant answers and related topics for queries using DuckDuckGo API"
    )
    parser.add_argument(
        "query",
        nargs=argparse.REMAINDER,
        help="Query to search for (e.g., 'python programming' or 'what is machine learning')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Join query parts into a single string
    query = " ".join(args.query)
    logger.debug(f"Parsed query: '{query}'")

    data = fetch_answer(query)

    # Check for search results in RelatedTopics or Results
    has_results = (
        data.get("AbstractText")
        or data.get("Answer")
        or data.get("Heading")
        or (data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0)
        or (data.get("Results") and len(data["Results"]) > 0)
    )
    logger.debug(f"Has results: {has_results}")

    if not has_results:
        logger.warning("No instant answers found for this query.")
        print(
            "Try a more specific query like 'python programming' or 'what is machine learning'"
        )
        return

    if args.json:
        logger.debug("Outputting results in JSON format")
        print(json.dumps(data, indent=2))
        return

    # Format human-readable output
    logger.debug("Formatting human-readable output")
    print(f"Search results for '{query}':")
    print("=" * 50)

    # Show main result if available
    if data.get("Heading"):
        logger.debug(f"Displaying main result: {data['Heading']}")
        print(f"{CYAN}Topic:{RESET} {data['Heading']}")
        if data.get("AbstractText"):
            abstract = data["AbstractText"]
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            print(f"{GREEN}Summary:{RESET} {abstract}")
        if data.get("AbstractURL"):
            print(f"{YELLOW}Source:{RESET} {data['AbstractURL']}")
        print()

    # Show direct answer if available
    if data.get("Answer"):
        logger.debug(f"Displaying direct answer: {data['Answer']}")
        answer = data["Answer"]
        if isinstance(answer, str) and answer:
            print(f"{GREEN}Answer:{RESET} {answer}")
        elif isinstance(answer, dict) and answer.get("result"):
            print(f"{GREEN}Answer:{RESET} {answer['result']}")
        print()

    # Show definition if available
    if data.get("Definition"):
        logger.debug(f"Displaying definition: {data['Definition']}")
        print(f"{GREEN}Definition:{RESET} {data['Definition']}")
        if data.get("DefinitionSource"):
            print(f"{YELLOW}Source:{RESET} {data['DefinitionSource']}")
        print()

    # Show search results from RelatedTopics
    related_topics = data.get("RelatedTopics", [])
    if related_topics:
        logger.debug(f"Displaying {len(related_topics)} related topics")
        print(f"{GREEN}Related Topics:{RESET}")
        for i, topic in enumerate(related_topics[:10]):  # Show up to 10 results
            if isinstance(topic, dict):
                text = topic.get("Text", "")
                url = topic.get("FirstURL", "")
                if text:
                    print(f"  {i+1}. {CYAN}{text}{RESET}")
                    if url:
                        print(f"     {YELLOW}URL:{RESET} {url}")
                    print()


if __name__ == "__main__":
    main()
