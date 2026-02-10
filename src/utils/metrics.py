"""
Metrics Collection and Monitoring
Tracks performance, usage, and quality metrics for the RAG system
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict, deque
from datetime import datetime
import json


class MetricsCollector:
    """
    Collects and aggregates metrics for the RAG system
    Tracks latency, accuracy, usage, and quality metrics
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            max_history: Maximum number of query records to keep
        """
        self.max_history = max_history
        
        # Metrics storage
        self.query_history = deque(maxlen=max_history)
        self.latencies = {
            'retrieval': deque(maxlen=max_history),
            'generation': deque(maxlen=max_history),
            'total': deque(maxlen=max_history)
        }
        self.relevance_scores = deque(maxlen=max_history)
        
        # Counters
        self.counters = defaultdict(int)
        
        # Start time
        self.start_time = datetime.now()
    
    def record_query(self, query: str, response: str, 
                    retrieval_time: float, generation_time: float,
                    relevance_scores: List[float], retrieved_chunks: int):
        """
        Record a complete query execution
        
        Args:
            query: User query
            response: Generated response
            retrieval_time: Time spent on retrieval (seconds)
            generation_time: Time spent on generation (seconds)
            relevance_scores: List of relevance scores for retrieved chunks
            retrieved_chunks: Number of chunks retrieved
        """
        total_time = retrieval_time + generation_time
        
        # Store query record
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_length': len(response),
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'chunks_retrieved': retrieved_chunks
        }
        self.query_history.append(record)
        
        # Update latencies
        self.latencies['retrieval'].append(retrieval_time)
        self.latencies['generation'].append(generation_time)
        self.latencies['total'].append(total_time)
        
        # Update relevance scores
        if relevance_scores:
            avg_score = sum(relevance_scores) / len(relevance_scores)
            self.relevance_scores.append(avg_score)
        
        # Increment counters
        self.counters['total_queries'] += 1
    
    def record_document_upload(self, filename: str, num_chunks: int, processing_time: float):
        """
        Record document upload metrics
        
        Args:
            filename: Name of uploaded file
            num_chunks: Number of chunks created
            processing_time: Time to process document
        """
        self.counters['total_documents'] += 1
        self.counters['total_chunks'] += num_chunks
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary of all metrics
        
        Returns:
            Dict containing all current metrics
        """
        return {
            'system': {
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'start_time': self.start_time.isoformat()
            },
            'counters': dict(self.counters),
            'latency': {
                'retrieval_ms': {
                    'avg': self._avg(self.latencies['retrieval']) * 1000,
                    'p50': self._percentile(self.latencies['retrieval'], 50) * 1000,
                    'p95': self._percentile(self.latencies['retrieval'], 95) * 1000,
                    'p99': self._percentile(self.latencies['retrieval'], 99) * 1000
                },
                'generation_ms': {
                    'avg': self._avg(self.latencies['generation']) * 1000,
                    'p50': self._percentile(self.latencies['generation'], 50) * 1000,
                    'p95': self._percentile(self.latencies['generation'], 95) * 1000,
                    'p99': self._percentile(self.latencies['generation'], 99) * 1000
                },
                'total_ms': {
                    'avg': self._avg(self.latencies['total']) * 1000,
                    'p50': self._percentile(self.latencies['total'], 50) * 1000,
                    'p95': self._percentile(self.latencies['total'], 95) * 1000,
                    'p99': self._percentile(self.latencies['total'], 99) * 1000
                }
            },
            'quality': {
                'avg_relevance_score': self._avg(self.relevance_scores),
                'relevance_above_80': self._count_above_threshold(self.relevance_scores, 0.8),
                'relevance_above_90': self._count_above_threshold(self.relevance_scores, 0.9)
            },
            'recent_queries': list(self.query_history)[-10:]  # Last 10 queries
        }
    
    def get_dashboard_metrics(self) -> Dict:
        """
        Get simplified metrics for dashboard display
        
        Returns:
            Dict with key metrics for UI display
        """
        return {
            'total_documents': self.counters.get('total_documents', 0),
            'total_queries': self.counters.get('total_queries', 0),
            'avg_latency_ms': round(self._avg(self.latencies['total']) * 1000, 1),
            'avg_relevance': round(self._avg(self.relevance_scores) * 100, 1)
        }
    
    def _avg(self, values: deque) -> float:
        """Calculate average of values"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _percentile(self, values: deque, p: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _count_above_threshold(self, values: deque, threshold: float) -> int:
        """Count values above threshold"""
        return sum(1 for v in values if v >= threshold)
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        metrics = self.get_metrics_summary()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def reset(self):
        """Reset all metrics"""
        self.query_history.clear()
        for key in self.latencies:
            self.latencies[key].clear()
        self.relevance_scores.clear()
        self.counters.clear()
        self.start_time = datetime.now()


class PerformanceTimer:
    """
    Context manager for timing operations
    
    Usage:
        with PerformanceTimer() as timer:
            # do something
        elapsed = timer.elapsed
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
