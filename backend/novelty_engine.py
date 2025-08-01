import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict

class EnhancedNoveltyEngine:
    """
    Enhanced novelty detection engine using TF-IDF and semantic similarity
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Add Korean stop words in production
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess Korean text"""
        # Remove special characters but keep Korean
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitting for Korean
        sentences = re.split(r'[.!?]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def calculate_novelty_score(
        self, 
        new_article: str, 
        read_articles: List[str]
    ) -> Tuple[float, List[Dict[str, any]]]:
        """
        Calculate novelty score and identify new information
        
        Returns:
            - novelty_score: 0-1 score indicating how much new info
            - new_info: List of new sentences with their novelty scores
        """
        if not read_articles:
            return 1.0, [{"text": "모든 내용이 새로운 정보입니다.", "score": 1.0}]
        
        # Preprocess all texts
        new_text_processed = self.preprocess_text(new_article)
        read_texts_processed = [self.preprocess_text(art) for art in read_articles]
        
        # Create document-term matrix
        all_texts = read_texts_processed + [new_text_processed]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        except:
            # Fallback if vectorization fails
            return 0.5, [{"text": "새로운 정보 분석 중 오류 발생", "score": 0.5}]
        
        # Calculate similarity between new article and each read article
        new_article_vector = tfidf_matrix[-1]
        similarities = []
        
        for i in range(len(read_articles)):
            sim = cosine_similarity(tfidf_matrix[i], new_article_vector)[0][0]
            similarities.append(sim)
        
        # Overall novelty score (inverse of max similarity)
        max_similarity = max(similarities) if similarities else 0
        novelty_score = 1 - max_similarity
        
        # Sentence-level novelty detection
        new_sentences = self.extract_sentences(new_article)
        read_sentences = []
        for article in read_articles:
            read_sentences.extend(self.extract_sentences(article))
        
        novel_sentences = []
        if new_sentences and read_sentences:
            # Vectorize all sentences
            all_sentences = read_sentences + new_sentences
            sentence_vectors = self.vectorizer.fit_transform(
                [self.preprocess_text(s) for s in all_sentences]
            )
            
            # Find novel sentences
            for i, new_sent in enumerate(new_sentences):
                new_sent_idx = len(read_sentences) + i
                new_sent_vector = sentence_vectors[new_sent_idx]
                
                # Calculate max similarity with read sentences
                max_sent_sim = 0
                for j in range(len(read_sentences)):
                    sim = cosine_similarity(
                        sentence_vectors[j], 
                        new_sent_vector
                    )[0][0]
                    max_sent_sim = max(max_sent_sim, sim)
                
                # If sentence is sufficiently novel
                if max_sent_sim < 0.7:  # Threshold for novelty
                    novel_sentences.append({
                        "text": new_sent,
                        "score": 1 - max_sent_sim,
                        "position": i
                    })
        
        # Sort by novelty score
        novel_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top 3 most novel sentences
        top_novel = novel_sentences[:3]
        
        if not top_novel:
            top_novel = [{
                "text": f"전체 내용의 {int((1-max_similarity)*100)}%가 새로운 정보입니다.",
                "score": novelty_score
            }]
        
        return novelty_score, top_novel
    
    def identify_highlight_regions(
        self, 
        article_text: str, 
        novel_sentences: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Identify regions in the article to highlight
        
        Returns list of regions with start/end positions
        """
        highlights = []
        
        for novel_sent in novel_sentences:
            sent_text = novel_sent['text']
            # Find position in original text
            start_pos = article_text.find(sent_text)
            if start_pos != -1:
                end_pos = start_pos + len(sent_text)
                highlights.append({
                    "start": start_pos,
                    "end": end_pos,
                    "text": sent_text,
                    "score": novel_sent['score'],
                    "reason": f"신규 정보 (유사도: {int((1-novel_sent['score'])*100)}%)"
                })
        
        return highlights


class ContentDuplicationDetector:
    """
    Detect and quantify content duplication between articles
    """
    
    def __init__(self):
        self.shingle_size = 3  # N-gram size for shingling
        
    def create_shingles(self, text: str, n: int = 3) -> set:
        """Create n-gram shingles from text"""
        words = text.lower().split()
        shingles = set()
        
        for i in range(len(words) - n + 1):
            shingle = ' '.join(words[i:i+n])
            shingles.add(shingle)
            
        return shingles
    
    def calculate_jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_duplication(
        self, 
        new_article: str, 
        read_articles: List[str]
    ) -> Dict[str, float]:
        """
        Detect content duplication
        
        Returns:
            - duplication_ratio: Overall duplication ratio
            - unique_content_ratio: Ratio of unique content
            - duplicated_sections: List of duplicated text sections
        """
        new_shingles = self.create_shingles(new_article, self.shingle_size)
        
        max_similarity = 0
        total_similarity = 0
        
        for read_article in read_articles:
            read_shingles = self.create_shingles(read_article, self.shingle_size)
            similarity = self.calculate_jaccard_similarity(new_shingles, read_shingles)
            max_similarity = max(max_similarity, similarity)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(read_articles) if read_articles else 0
        
        return {
            "duplication_ratio": max_similarity,
            "unique_content_ratio": 1 - max_similarity,
            "average_similarity": avg_similarity,
            "is_duplicate": max_similarity > 0.8  # Threshold for duplicate
        }


# Integration with main FastAPI app
def integrate_enhanced_novelty(app):
    """Add enhanced novelty detection to existing FastAPI app"""
    
    novelty_engine = EnhancedNoveltyEngine()
    dup_detector = ContentDuplicationDetector()
    
    @app.post("/api/news/analyze-novelty")
    async def analyze_novelty(
        article_id: str,
        user_id: str,
        article_content: str,
        read_article_ids: List[str]
    ):
        """Analyze novelty of an article compared to user's reading history"""
        
        # Get read articles content (from database in production)
        read_contents = []  # Fetch from DB
        
        # Calculate novelty
        novelty_score, novel_info = novelty_engine.calculate_novelty_score(
            article_content, 
            read_contents
        )
        
        # Detect duplication
        dup_info = dup_detector.detect_duplication(
            article_content,
            read_contents
        )
        
        # Get highlight regions
        highlights = novelty_engine.identify_highlight_regions(
            article_content,
            novel_info
        )
        
        return {
            "article_id": article_id,
            "novelty_score": novelty_score,
            "novel_sentences": novel_info,
            "duplication_info": dup_info,
            "highlight_regions": highlights
        }