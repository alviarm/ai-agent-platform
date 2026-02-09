"""Feedback reporting and aggregation."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import DATE_FORMAT, TREND_WINDOW_DAYS, TOP_KEYWORDS_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackReporter:
    """Generate reports from analyzed feedback."""
    
    def __init__(self, analysis_results: List[Dict] = None):
        """Initialize reporter.
        
        Args:
            analysis_results: List of analyzed feedback results
        """
        self.analysis_results = analysis_results or []
    
    def add_results(self, results: List[Dict]):
        """Add analysis results."""
        self.analysis_results.extend(results)
    
    def generate_summary(self) -> Dict:
        """Generate overall summary statistics.
        
        Returns:
            Summary dictionary
        """
        if not self.analysis_results:
            return {}
        
        total = len(self.analysis_results)
        
        # Sentiment distribution
        sentiments = [r["sentiment"]["label"] for r in self.analysis_results]
        sentiment_counts = defaultdict(int)
        for s in sentiments:
            sentiment_counts[s] += 1
        
        # Average confidence
        confidences = [r["sentiment"]["confidence"] for r in self.analysis_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # All keywords
        all_keywords = []
        for result in self.analysis_results:
            for kw in result.get("keywords", []):
                all_keywords.append((kw["phrase"], kw["score"]))
        
        # Aggregate keyword scores
        keyword_scores = defaultdict(float)
        keyword_counts = defaultdict(int)
        for phrase, score in all_keywords:
            keyword_scores[phrase] += score
            keyword_counts[phrase] += 1
        
        # Average keyword scores
        top_keywords = [
            {"phrase": phrase, "score": keyword_scores[phrase] / keyword_counts[phrase]}
            for phrase in keyword_scores
        ]
        top_keywords.sort(key=lambda x: x["score"], reverse=True)
        
        # Topic distribution
        topic_counts = defaultdict(int)
        for result in self.analysis_results:
            topics = result.get("topics", [])
            if topics:
                # Take top topic
                topic_counts[topics[0]["topic_id"]] += 1
        
        return {
            "total_feedback": total,
            "sentiment_distribution": dict(sentiment_counts),
            "sentiment_percentages": {
                k: round(v / total * 100, 2) if total > 0 else 0
                for k, v in sentiment_counts.items()
            },
            "average_confidence": round(avg_confidence, 3),
            "top_keywords": top_keywords[:TOP_KEYWORDS_COUNT],
            "topic_distribution": dict(topic_counts),
        }
    
    def generate_trend_report(
        self,
        date_field: str = "timestamp",
        window_days: int = TREND_WINDOW_DAYS,
    ) -> List[Dict]:
        """Generate trend report over time.
        
        Args:
            date_field: Field name for date in metadata
            window_days: Aggregation window in days
            
        Returns:
            List of trend data points
        """
        if not self.analysis_results:
            return []
        
        # Group by date window
        windows = defaultdict(list)
        
        for result in self.analysis_results:
            metadata = result.get("metadata", {})
            date_str = metadata.get(date_field)
            
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    # Round to window
                    window_start = date - timedelta(days=date.weekday())
                    window_key = window_start.strftime(DATE_FORMAT)
                    windows[window_key].append(result)
                except:
                    pass
        
        # Calculate metrics per window
        trends = []
        for window_key in sorted(windows.keys()):
            window_results = windows[window_key]
            
            # Sentiment distribution
            sentiments = [r["sentiment"]["label"] for r in window_results]
            sentiment_counts = defaultdict(int)
            for s in sentiments:
                sentiment_counts[s] += 1
            
            total = len(window_results)
            
            trends.append({
                "period": window_key,
                "total_feedback": total,
                "sentiment": dict(sentiment_counts),
                "positive_rate": round(sentiment_counts.get("positive", 0) / total * 100, 2) if total > 0 else 0,
                "negative_rate": round(sentiment_counts.get("negative", 0) / total * 100, 2) if total > 0 else 0,
            })
        
        return trends
    
    def generate_intent_report(self) -> Dict:
        """Generate report grouped by intent.
        
        Returns:
            Intent-based report dictionary
        """
        if not self.analysis_results:
            return {}
        
        # Group by intent
        by_intent = defaultdict(list)
        for result in self.analysis_results:
            intent = result.get("metadata", {}).get("intent", "unknown")
            by_intent[intent].append(result)
        
        intent_reports = {}
        for intent, results in by_intent.items():
            total = len(results)
            
            # Sentiment for this intent
            sentiments = [r["sentiment"]["label"] for r in results]
            sentiment_counts = defaultdict(int)
            for s in sentiments:
                sentiment_counts[s] += 1
            
            # Top keywords for this intent
            keywords = []
            for r in results:
                keywords.extend([kw["phrase"] for kw in r.get("keywords", [])])
            keyword_freq = defaultdict(int)
            for kw in keywords:
                keyword_freq[kw] += 1
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            intent_reports[intent] = {
                "total": total,
                "sentiment_distribution": dict(sentiment_counts),
                "top_keywords": [{"phrase": k, "count": c} for k, c in top_keywords],
                "satisfaction_score": round(
                    sentiment_counts.get("positive", 0) / total * 100, 2
                ) if total > 0 else 0,
            }
        
        return intent_reports
    
    def export_to_json(self, output_path: str, include_raw: bool = False):
        """Export report to JSON.
        
        Args:
            output_path: Output file path
            include_raw: Whether to include raw analysis results
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.generate_summary(),
            "trends": self.generate_trend_report(),
            "by_intent": self.generate_intent_report(),
        }
        
        if include_raw:
            report["raw_results"] = self.analysis_results
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to {output_path}")
    
    def export_to_csv(self, output_dir: str):
        """Export data to CSV files.
        
        Args:
            output_dir: Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Main feedback data
        rows = []
        for result in self.analysis_results:
            row = {
                "text": result.get("original_text", ""),
                "processed_text": result.get("processed_text", ""),
                "sentiment": result.get("sentiment", {}).get("label", ""),
                "sentiment_confidence": result.get("sentiment", {}).get("confidence", 0),
            }
            # Add metadata
            row.update(result.get("metadata", {}))
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/feedback_data.csv", index=False)
        
        # Keywords data
        keyword_rows = []
        for result in self.analysis_results:
            for kw in result.get("keywords", []):
                keyword_rows.append({
                    "feedback_id": result.get("metadata", {}).get("id", ""),
                    "keyword": kw["phrase"],
                    "score": kw["score"],
                })
        
        if keyword_rows:
            pd.DataFrame(keyword_rows).to_csv(
                f"{output_dir}/keywords.csv", index=False
            )
        
        logger.info(f"CSV files exported to {output_dir}")
    
    def generate_weekly_report(
        self,
        output_path: str = None,
    ) -> str:
        """Generate a formatted weekly report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        summary = self.generate_summary()
        trends = self.generate_trend_report()
        by_intent = self.generate_intent_report()
        
        lines = [
            "=" * 60,
            "WEEKLY CUSTOMER FEEDBACK REPORT",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Feedback: {summary.get('total_feedback', 0)}",
            f"Average Confidence: {summary.get('average_confidence', 0):.3f}",
            "",
            "Sentiment Distribution:",
        ]
        
        for sentiment, count in summary.get("sentiment_distribution", {}).items():
            pct = summary.get("sentiment_percentages", {}).get(sentiment, 0)
            lines.append(f"  {sentiment.capitalize()}: {count} ({pct}%)")
        
        lines.extend([
            "",
            "TOP KEYWORDS",
            "-" * 40,
        ])
        
        for kw in summary.get("top_keywords", [])[:10]:
            lines.append(f"  - {kw['phrase']}: {kw['score']:.3f}")
        
        lines.extend([
            "",
            "BY INTENT",
            "-" * 40,
        ])
        
        for intent, data in by_intent.items():
            lines.extend([
                f"\n{intent.upper()}:",
                f"  Total: {data['total']}",
                f"  Satisfaction: {data['satisfaction_score']}%",
            ])
        
        lines.extend([
            "",
            "TRENDS",
            "-" * 40,
        ])
        
        for trend in trends[-4:]:  # Last 4 periods
            lines.append(
                f"  {trend['period']}: {trend['total_feedback']} feedback, "
                f"{trend['positive_rate']}% positive"
            )
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Weekly report saved to {output_path}")
        
        return report
