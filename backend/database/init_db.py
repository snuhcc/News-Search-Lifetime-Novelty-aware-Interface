# database/init_db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

def init_database(database_url: str):
    """Initialize database with tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()