import argparse
import asyncio
from pathlib import Path

from .config import get_settings
from .knowledge import KnowledgeBase


async def run(path: Path, source: str, append: bool, page_summary: bool) -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings)
    kb.init_db()
    text = path.read_text(encoding="utf-8")
    if page_summary:
        result = await kb.ingest_page_summary_document(text, source, not append)
    else:
        result = await kb.ingest_text(text, source, not append)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a UTF-8 text file into the restaurant RAG knowledge base.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", default="restaurant-kb")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--page-summary", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.path, args.source, args.append, args.page_summary))


if __name__ == "__main__":
    main()
