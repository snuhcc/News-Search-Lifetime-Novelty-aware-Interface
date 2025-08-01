# database/crud.py
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from . import models

class ArticleCRUD:
    @staticmethod
    def get_articles_by_keyword(
        db: Session, 
        keyword: str, 
        limit: int = 50
    ) -> List[models.Article]:
        """Get articles matching keyword"""
        return db.query(models.Article).filter(
            models.Article.title.contains(keyword) | 
            models.Article.content.contains(keyword)
        ).order_by(models.Article.publish_time.desc()).limit(limit).all()
    
    @staticmethod
    def get_articles_by_category(
        db: Session,
        category: str,
        since: Optional[datetime] = None
    ) -> List[models.Article]:
        """Get articles by category"""
        query = db.query(models.Article).filter(
            models.Article.category == category
        )
        if since:
            query = query.filter(models.Article.publish_time >= since)
        return query.order_by(models.Article.publish_time.desc()).all()


class UserCRUD:
    @staticmethod
    def get_user_clicks(
        db: Session,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[models.UserClick]:
        """Get user's click history"""
        query = db.query(models.UserClick).filter(
            models.UserClick.user_id == user_id
        ).order_by(models.UserClick.click_time.desc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @staticmethod
    def calculate_user_topic_lifetime(
        db: Session,
        user_id: str,
        topic: str
    ) -> float:
        """Calculate user's lifetime for a topic"""
        # Get clicks for this topic
        clicks = db.query(models.UserClick).join(
            models.Article
        ).filter(
            models.UserClick.user_id == user_id,
            models.Article.category == topic
        ).order_by(models.UserClick.click_time).all()
        
        if len(clicks) < 2:
            # Default lifetimes
            defaults = {
                "스포츠": 10,
                "경제": 72,
                "정치": 48,
                "문화": 168,
                "IT": 48
            }
            return defaults.get(topic, 48)
        
        # Calculate intervals
        intervals = []
        for click in clicks:
            time_since_publish = (
                click.click_time - click.article.publish_time
            ).total_seconds() / 3600
            intervals.append(time_since_publish)
        
        # Use 90th percentile
        import numpy as np
        return float(np.percentile(intervals, 90))
    
    @staticmethod
    def update_user_topic_lifetime(
        db: Session,
        user_id: str,
        topic: str,
        lifetime_hours: float
    ):
        """Update or create user topic lifetime"""
        lifetime = db.query(models.UserTopicLifetime).filter(
            models.UserTopicLifetime.user_id == user_id,
            models.UserTopicLifetime.topic == topic
        ).first()
        
        if lifetime:
            lifetime.lifetime_hours = lifetime_hours
            lifetime.last_calculated = datetime.utcnow()
        else:
            lifetime = models.UserTopicLifetime(
                user_id=user_id,
                topic=topic,
                lifetime_hours=lifetime_hours
            )
            db.add(lifetime)
        
        db.commit()
        return lifetime