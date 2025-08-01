from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager
from sample_news_data import get_sample_news_data

# Data Models
class NewsArticle(BaseModel):
    id: str
    title: str
    content: str
    category: str
    subcategory: Optional[str] = None
    publish_time: datetime
    source: str
    summary: Optional[str] = None
    url: Optional[str] = None  # 실제 뉴스 링크 URL

class UserClickHistory(BaseModel):
    user_id: str
    article_id: str
    click_time: datetime
    reading_duration: Optional[int] = None  # in seconds

class UserTopicLifetime(BaseModel):
    user_id: str
    topic: str
    lifetime_hours: float

class NewsRecommendation(BaseModel):
    article: NewsArticle
    score: float
    lifetime_explanation: Optional[str] = None
    novelty_score: float
    new_content_summary: Optional[List[str]] = None
    is_valid: bool  # Based on lifetime
    has_new_info: bool

# In-memory storage (replace with database in production)
articles_db: Dict[str, NewsArticle] = {}
user_history: Dict[str, List[UserClickHistory]] = defaultdict(list)
user_topic_lifetimes: Dict[str, Dict[str, float]] = defaultdict(dict)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    sample_articles = get_sample_news_data()
    
    for article_data in sample_articles:
        article = NewsArticle(**article_data)
        articles_db[article.id] = article
    
    print(f"✅ {len(sample_articles)}개의 뉴스 데이터가 로드되었습니다.")
    
    yield
    
    # Shutdown (if needed)
    # Clean up resources here

# Create FastAPI app with lifespan
app = FastAPI(title="NLNI News Recommendation System", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifetime Engine
class LifetimeEngine:
    @staticmethod
    def calculate_user_topic_lifetime(user_id: str, topic: str, history: List[UserClickHistory]) -> float:
        """Calculate user-specific lifetime for a topic based on historical behavior"""
        topic_clicks = [h for h in history if articles_db.get(h.article_id) and 
                       articles_db[h.article_id].category == topic]
        
        if len(topic_clicks) < 2:
            # Default lifetimes by topic
            default_lifetimes = {
                "스포츠": 10,  # 10 hours
                "날씨": 24,    # 24 hours
                "여행": 168,   # 1 week
                "경제": 72,    # 3 days
                "정치": 48,    # 2 days
                "IT": 48,      # 2 days
                "문화": 168,   # 1 week
                "사회": 48,    # 2 days
            }
            return default_lifetimes.get(topic, 48)
        
        # Calculate based on user's click patterns
        click_intervals = []
        sorted_clicks = sorted(topic_clicks, key=lambda x: x.click_time)
        
        for i in range(1, len(sorted_clicks)):
            article = articles_db[sorted_clicks[i].article_id]
            time_since_publish = (sorted_clicks[i].click_time - article.publish_time).total_seconds() / 3600
            click_intervals.append(time_since_publish)
        
        # 90th percentile as lifetime
        if click_intervals:
            return np.percentile(click_intervals, 90)
        return 48  # Default 48 hours

    @staticmethod
    def is_article_valid(article: NewsArticle, user_topic_lifetime: float) -> bool:
        """Check if article is still valid based on lifetime"""
        age_hours = (datetime.now() - article.publish_time).total_seconds() / 3600
        return age_hours <= user_topic_lifetime

# Novelty Engine
class NoveltyEngine:
    @staticmethod
    def calculate_novelty_score(article: NewsArticle, user_history: List[UserClickHistory]) -> tuple[float, List[str]]:
        """Calculate novelty score and extract new information"""
        # Get articles from same topic that user has read
        read_articles = []
        for click in user_history:
            if click.article_id in articles_db:
                hist_article = articles_db[click.article_id]
                if hist_article.category == article.category:
                    read_articles.append(hist_article)
        
        if not read_articles:
            return 1.0, ["모든 내용이 새로운 정보입니다."]
        
        # Simple novelty detection (in production, use NLP models)
        # For demo, we'll simulate novelty detection
        novelty_score = max(0.3, 1.0 - (0.2 * len(read_articles)))
        
        new_info = []
        if novelty_score > 0.5:
            new_info.append(f"이전 기사 대비 {int(novelty_score * 100)}%의 새로운 정보를 포함합니다.")
        
        return novelty_score, new_info

# API Endpoints
@app.get("/api/news/search")
async def search_news(
    keyword: str = Query(..., description="검색 키워드"),
    sort: str = Query("relevance", description="정렬 방식: relevance, latest, lifetime, novelty"),
    user_id: str = Query(..., description="사용자 ID"),
    lifetime_filter: Optional[float] = Query(None, description="관심 지속 기간 필터 (시간)"),
    novelty_filter: Optional[bool] = Query(None, description="새로운 정보만 표시")
):
    """뉴스 검색 및 추천"""
    # Get user's history
    user_clicks = user_history.get(user_id, [])
    
    # Filter articles by keyword
    matching_articles = [
        article for article in articles_db.values()
        if keyword.lower() in article.title.lower() or keyword.lower() in article.content.lower()
    ]
    
    recommendations = []
    for article in matching_articles:
        # Calculate lifetime
        lifetime = LifetimeEngine.calculate_user_topic_lifetime(user_id, article.category, user_clicks)
        is_valid = LifetimeEngine.is_article_valid(article, lifetime)
        
        # Calculate novelty
        novelty_score, new_info = NoveltyEngine.calculate_novelty_score(article, user_clicks)
        
        # Generate lifetime explanation
        age_hours = (datetime.now() - article.publish_time).total_seconds() / 3600
        lifetime_explanation = None
        if is_valid:
            lifetime_explanation = f"이 기사는 {int(age_hours)}시간 전 발행되었지만, '{article.category}' 주제에 대한 귀하의 관심은 평균 {int(lifetime)}시간 지속됩니다."
        
        # Apply filters
        if lifetime_filter and age_hours > lifetime_filter:
            continue
        if novelty_filter and novelty_score < 0.5:
            continue
        
        recommendation = NewsRecommendation(
            article=article,
            score=0.8 if is_valid else 0.3,  # Simple scoring
            lifetime_explanation=lifetime_explanation,
            novelty_score=novelty_score,
            new_content_summary=new_info if novelty_score > 0.5 else None,
            is_valid=is_valid,
            has_new_info=novelty_score > 0.5
        )
        recommendations.append(recommendation)
    
    # Sort based on criteria
    if sort == "latest":
        recommendations.sort(key=lambda x: x.article.publish_time, reverse=True)
    elif sort == "lifetime":
        recommendations.sort(key=lambda x: (x.is_valid, x.score), reverse=True)
    elif sort == "novelty":
        recommendations.sort(key=lambda x: x.novelty_score, reverse=True)
    else:  # relevance
        recommendations.sort(key=lambda x: x.score, reverse=True)
    
    return recommendations

@app.get("/api/news/timeline/{keyword}")
async def get_timeline(keyword: str, days: int = Query(7, description="타임라인 기간")):
    """키워드에 대한 주요 이벤트 타임라인"""
    # Filter articles by keyword and time range
    cutoff_date = datetime.now() - timedelta(days=days)
    timeline_articles = [
        article for article in articles_db.values()
        if (keyword.lower() in article.title.lower() or keyword.lower() in article.content.lower())
        and article.publish_time >= cutoff_date
    ]
    
    # Sort by time
    timeline_articles.sort(key=lambda x: x.publish_time)
    
    # Create timeline events
    timeline_events = []
    for article in timeline_articles:
        event = {
            "timestamp": article.publish_time.isoformat(),
            "title": article.title,
            "type": "주요" if any(word in article.title for word in ["발표", "공식", "확정"]) else "일반",
            "article_id": article.id
        }
        timeline_events.append(event)
    
    return {"keyword": keyword, "events": timeline_events}

@app.post("/api/user/click")
async def record_click(click: UserClickHistory):
    """사용자 클릭 기록"""
    user_history[click.user_id].append(click)
    return {"status": "recorded"}

@app.put("/api/user/preferences")
async def update_preferences(
    user_id: str,
    topic: str,
    lifetime_hours: float
):
    """사용자 주제별 관심 지속 기간 설정"""
    user_topic_lifetimes[user_id][topic] = lifetime_hours
    return {"status": "updated"}

@app.get("/api/news/{article_id}/highlights")
async def get_article_highlights(article_id: str, user_id: str):
    """기사 내 새로운 정보 하이라이트"""
    if article_id not in articles_db:
        raise HTTPException(status_code=404, detail="Article not found")
    
    article = articles_db[article_id]
    user_clicks = user_history.get(user_id, [])
    
    # Simulate highlighting new content
    # In production, use NLP to identify actual new sentences
    highlights = {
        "article_id": article_id,
        "highlighted_sections": [
            {
                "start": 50,
                "end": 150,
                "text": "새로운 정보가 포함된 부분입니다.",
                "reason": "이전 기사에서 다루지 않은 내용"
            }
        ]
    }
    
    return highlights

# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)