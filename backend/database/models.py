# database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    clicks = relationship("UserClick", back_populates="user")
    topic_lifetimes = relationship("UserTopicLifetime", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")


class Article(Base):
    __tablename__ = "articles"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    category = Column(String, nullable=False, index=True)
    subcategory = Column(String)
    source = Column(String, nullable=False)
    author = Column(String)
    publish_time = Column(DateTime, nullable=False, index=True)
    
    # Computed fields
    word_count = Column(Integer)
    reading_time_minutes = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    clicks = relationship("UserClick", back_populates="article")
    similarities = relationship("ArticleSimilarity", 
                              foreign_keys="ArticleSimilarity.article1_id",
                              back_populates="article1")


class UserClick(Base):
    __tablename__ = "user_clicks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    article_id = Column(String, ForeignKey("articles.id"), nullable=False)
    click_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Engagement metrics
    reading_duration_seconds = Column(Integer)
    scroll_depth_percentage = Column(Float)
    shared = Column(Boolean, default=False)
    bookmarked = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="clicks")
    article = relationship("Article", back_populates="clicks")


class UserTopicLifetime(Base):
    __tablename__ = "user_topic_lifetimes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    topic = Column(String, nullable=False)
    lifetime_hours = Column(Float, nullable=False)
    
    # Statistics
    click_count = Column(Integer, default=0)
    last_calculated = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float, default=0.5)  # 0-1, based on data amount
    
    # Relationships
    user = relationship("User", back_populates="topic_lifetimes")
    
    # Unique constraint
    __table_args__ = (
        {"extend_existing": True, "sqlite_autoincrement": True},
    )


class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Filter preferences
    novelty_threshold = Column(Float, default=0.5)  # 0-1
    duplicate_tolerance = Column(String, default="medium")  # low, medium, high
    
    # Display preferences
    highlights_enabled = Column(Boolean, default=True)
    timeline_enabled = Column(Boolean, default=True)
    
    # Notification preferences
    breaking_news_alerts = Column(Boolean, default=False)
    topic_update_alerts = Column(JSON, default={})  # {"topic": bool}
    
    # Relationships
    user = relationship("User", back_populates="preferences")


class ArticleSimilarity(Base):
    __tablename__ = "article_similarities"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article1_id = Column(String, ForeignKey("articles.id"), nullable=False)
    article2_id = Column(String, ForeignKey("articles.id"), nullable=False)
    
    # Similarity metrics
    content_similarity = Column(Float)  # 0-1
    title_similarity = Column(Float)  # 0-1
    category_match = Column(Boolean)
    
    # Novelty metrics
    unique_content_ratio = Column(Float)  # 0-1, how much of article2 is new
    novel_sentences = Column(JSON)  # List of novel sentences
    
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article1 = relationship("Article", foreign_keys=[article1_id])
    article2 = relationship("Article", foreign_keys=[article2_id])


class TimelineEvent(Base):
    __tablename__ = "timeline_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String, nullable=False, index=True)
    article_id = Column(String, ForeignKey("articles.id"), nullable=False)
    
    event_time = Column(DateTime, nullable=False)
    event_type = Column(String)  # "major", "breaking", "update", "general"
    importance_score = Column(Float, default=0.5)
    
    # Event metadata
    title = Column(String)
    summary = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)