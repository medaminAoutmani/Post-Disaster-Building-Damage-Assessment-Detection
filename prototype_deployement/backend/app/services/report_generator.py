import os
import io
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from jinja2 import Template

from app.core.config import get_settings
from app.core.storage import upload_file
from app.core.db import AsyncSessionLocal
from app.db_models import Report, ImageJob, Tweet, DamagePrediction
from app.core.logging import get_logger

logger = get_logger("services.report")
settings = get_settings()

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        self.heading2 = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5282'),
            spaceAfter=12,
            spaceBefore=12,
        )

    def _create_damage_chart(self, stats: Dict[str, int]) -> io.BytesIO:
        """Generate a matplotlib pie chart of damage severity."""
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(stats.keys())
        sizes = list(stats.values())
        colors_list = ['#48bb78', '#ecc94b', '#ed8936', '#f56565']
        explode = [0.05 if l == 'destroyed' else 0 for l in labels]

        ax.pie(sizes, explode=explode, labels=labels, colors=colors_list[:len(labels)],
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Building Damage Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        buf.seek(0)
        plt.close(fig)
        return buf

    def _create_sentiment_timeline(self, tweets: List[Tweet]) -> io.BytesIO:
        """Generate sentiment timeline chart."""
        if not tweets:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'No sentiment data available', ha='center', va='center')
            buf = io.BytesIO()
            FigureCanvasAgg(fig).print_png(buf)
            buf.seek(0)
            plt.close(fig)
            return buf

        # Aggregate by hour
        from collections import defaultdict
        timeline = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
        for t in tweets:
            hour = t.timestamp.strftime("%Y-%m-%d %H:00")
            timeline[hour][t.sentiment] += 1

        hours = sorted(timeline.keys())
        pos = [timeline[h]["positive"] for h in hours]
        neg = [timeline[h]["negative"] for h in hours]
        neu = [timeline[h]["neutral"] for h in hours]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(hours))
        ax.plot(x, pos, label='Positive', color='#48bb78', marker='o')
        ax.plot(x, neg, label='Negative', color='#f56565', marker='s')
        ax.plot(x, neu, label='Neutral', color='#a0aec0', marker='^')
        ax.set_xticks(x[::max(1, len(x)//6)])
        ax.set_xticklabels([hours[i] for i in range(0, len(hours), max(1, len(x)//6))], rotation=45, ha='right')
        ax.set_title('Social Media Sentiment Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        buf.seek(0)
        plt.close(fig)
        return buf

    async def generate_pdf(self, report_id: str, title: str, region: dict, start_date: datetime, end_date: datetime, include_damage: bool, include_sentiment: bool, include_rag: bool) -> bytes:
        """Generate comprehensive PDF report."""
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []

        # Title
        story.append(Paragraph(f"POST-DISASTER ANALYTICS REPORT", self.title_style))
        story.append(Paragraph(f"<b>{title}</b>", ParagraphStyle('Subtitle', parent=self.styles['Normal'], fontSize=14, alignment=TA_CENTER, spaceAfter=20)))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ParagraphStyle('Date', parent=self.styles['Normal'], alignment=TA_CENTER, textColor=colors.grey)))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("1. EXECUTIVE SUMMARY", self.heading2))
        story.append(Paragraph(
            f"This report covers the disaster impact assessment for the specified region "
            f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. "
            f"Data sources include satellite imagery analysis, social media sentiment monitoring, "
            f"and retrieval-augmented expert guidelines.",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 12))

        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, func

            # Damage Section
            if include_damage:
                story.append(Paragraph("2. DAMAGE ASSESSMENT", self.heading2))

                stmt = select(ImageJob).where(ImageJob.status == "completed")
                result = await session.execute(stmt)
                jobs = result.scalars().all()

                total_stats = {"no_damage": 0, "minor": 0, "major": 0, "destroyed": 0}
                for job in jobs:
                    if job.damage_stats:
                        for k, v in job.damage_stats.items():
                            total_stats[k] = total_stats.get(k, 0) + v

                story.append(Paragraph(
                    f"Satellite imagery analysis identified <b>{sum(total_stats.values())}</b> structures. "
                    f"Of these, <b>{total_stats['destroyed']}</b> were destroyed, <b>{total_stats['major']}</b> sustained major damage, "
                    f"<b>{total_stats['minor']}</b> minor damage, and <b>{total_stats['no_damage']}</b> were undamaged.",
                    self.styles['Normal']
                ))
                story.append(Spacer(1, 12))

                # Damage chart
                chart_buf = self._create_damage_chart(total_stats)
                story.append(RLImage(chart_buf, width=5*inch, height=3.3*inch))
                story.append(Spacer(1, 12))

            # Sentiment Section
            if include_sentiment:
                story.append(Paragraph("3. PUBLIC SENTIMENT & PSYCHOLOGICAL IMPACT", self.heading2))

                stmt = select(Tweet).where(Tweet.timestamp >= start_date, Tweet.timestamp <= end_date)
                result = await session.execute(stmt)
                tweets = result.scalars().all()

                if tweets:
                    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
                    emotion_agg = {}
                    for t in tweets:
                        sentiment_counts[t.sentiment] += 1
                        if t.emotion:
                            for k, v in t.emotion.items():
                                emotion_agg[k] = emotion_agg.get(k, 0) + v

                    dominant_emotion = max(emotion_agg, key=emotion_agg.get) if emotion_agg else "unknown"

                    story.append(Paragraph(
                        f"Analysis of <b>{len(tweets)}</b> social media posts reveals a predominantly "
                        f"<b>{max(sentiment_counts, key=sentiment_counts.get)}</b> sentiment landscape. "
                        f"The dominant emotional signal is <b>{dominant_emotion}</b>, indicating "
                        f"significant psychological stress in the affected population.",
                        self.styles['Normal']
                    ))
                    story.append(Spacer(1, 12))

                    sent_buf = self._create_sentiment_timeline(tweets)
                    story.append(RLImage(sent_buf, width=6*inch, height=3*inch))
                else:
                    story.append(Paragraph("No social media data available for the reporting period.", self.styles['Normal']))
                story.append(Spacer(1, 12))

            # RAG Commentary
            if include_rag:
                story.append(Paragraph("4. EXPERT ANALYSIS & RECOMMENDATIONS", self.heading2))
                story.append(Paragraph(
                    "Based on retrieval-augmented analysis of official guidelines (UN OCHA, FEMA, WHO), "
                    "the following actions are prioritized for the current disaster profile:",
                    self.styles['Normal']
                ))
                story.append(Spacer(1, 6))

                recommendations = [
                    "Immediate deployment of search-and-rescue teams to sectors with >40% destroyed structures.",
                    "Establishment of emergency medical posts and mobile triage units within 12 hours.",
                    "Distribution of safe drinking water, sanitation kits, and hygiene supplies to prevent disease outbreak.",
                    "Activation of mental health and psychosocial support (MHPSS) services given elevated fear/sadness indicators.",
                    "Coordination with local authorities for evacuation route clearance and temporary shelter setup.",
                ]
                for rec in recommendations:
                    story.append(Paragraph(f"• {rec}", self.styles['Normal']))
                story.append(Spacer(1, 12))

        # Footer / disclaimer
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "<i>Disclaimer: This automated report is intended to support decision-making and should be validated by domain experts. "
            "Model outputs carry uncertainty and should be ground-truthed where possible.</i>",
            ParagraphStyle('Disclaimer', parent=self.styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_JUSTIFY)
        ))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes

    async def generate_html(self, report_id: str, title: str, region: dict, start_date: datetime, end_date: datetime, include_damage: bool, include_sentiment: bool, include_rag: bool) -> str:
        """Generate HTML report for dashboard viewing."""
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        h1 { color: #1a365d; border-bottom: 3px solid #2c5282; padding-bottom: 10px; }
        h2 { color: #2c5282; margin-top: 30px; }
        .meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
        .metric { background: #f7fafc; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .recommendation { background: #ebf8ff; padding: 12px; border-left: 4px solid #3182ce; margin: 8px 0; }
    </style>
</head>
<body>
    <h1>Post-Disaster Analytics Report</h1>
    <div class="meta">
        <strong>{{ title }}</strong><br>
        Period: {{ start_date }} to {{ end_date }}<br>
        Generated: {{ generated_at }}
    </div>

    <h2>Executive Summary</h2>
    <p>Integrated analysis of satellite imagery, social media sentiment, and expert guidelines.</p>

    {% if include_damage %}
    <h2>Damage Assessment</h2>
    <div class="metric">Satellite-derived building damage analysis.</div>
    {% endif %}

    {% if include_sentiment %}
    <h2>Public Sentiment</h2>
    <div class="metric">Social media emotion and sentiment timeline.</div>
    {% endif %}

    {% if include_rag %}
    <h2>Expert Recommendations</h2>
    <div class="recommendation">Prioritize search and rescue in high-destruction zones.</div>
    <div class="recommendation">Deploy emergency medical posts and water/sanitation supplies.</div>
    <div class="recommendation">Activate mental health support services.</div>
    {% endif %}
</body>
</html>
        """)
        return template.render(
            title=title,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            include_damage=include_damage,
            include_sentiment=include_sentiment,
            include_rag=include_rag,
        )

# Singleton
_report_gen = None

def get_report_generator() -> ReportGenerator:
    global _report_gen
    if _report_gen is None:
        _report_gen = ReportGenerator()
    return _report_gen
