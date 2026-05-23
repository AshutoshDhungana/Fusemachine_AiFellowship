import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/mydb",
)

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    # Quick connectivity check at startup
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection established successfully.")
except Exception as exc:
    logger.error(f"Failed to connect to database: {exc}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a database session and
    guarantees it is closed afterwards — even on error.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        logger.debug("Database session closed.")
