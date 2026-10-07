import asyncio
from pathlib import Path

from .config import get_settings
from .knowledge import KnowledgeBase

DEFAULT_KB_PATH = Path("data/McDonalds_About_Our_Food_Summary.txt")
DEFAULT_SOURCE = "mcdonalds-about-our-food-2026-10-06"


async def run() -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings)
    kb.init_db()
    text = DEFAULT_KB_PATH.read_text(encoding="utf-8")
    result = await kb.ingest_page_summary_document(
        text=text,
        source=DEFAULT_SOURCE,
        replace_source=True,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(run())
