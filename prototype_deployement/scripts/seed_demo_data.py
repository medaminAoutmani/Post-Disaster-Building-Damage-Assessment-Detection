#!/usr/bin/env python3
"""
Seed script to populate the platform with demo data for testing.
Run after `make up` to initialize the database and services.
"""
import asyncio
import datetime
import json
import requests

BASE = "http://localhost:8000"

def seed_tweets():
    tweets = [
        {"tweet_id": "t001", "text": "City center is completely flooded! We need help urgently.", "user_location": "Las Vegas, NV", "timestamp": "2026-05-28T14:00:00Z"},
        {"tweet_id": "t002", "text": "Rescue teams have arrived downtown. Thank you everyone!", "user_location": "Las Vegas, NV", "timestamp": "2026-05-28T16:30:00Z"},
        {"tweet_id": "t003", "text": "Lost my home in the flood. Heartbroken.", "user_location": "North Las Vegas", "timestamp": "2026-05-28T18:00:00Z"},
        {"tweet_id": "t004", "text": "Water contamination reported in sector 4. Avoid tap water.", "user_location": "Las Vegas, NV", "timestamp": "2026-05-29T08:00:00Z"},
        {"tweet_id": "t005", "text": "Emergency shelter setup at Convention Center. Safe and dry.", "user_location": "Las Vegas, NV", "timestamp": "2026-05-29T10:00:00Z"},
    ]
    for t in tweets:
        try:
            r = requests.post(f"{BASE}/ingest/tweet", json=t)
            print(f"Tweet {t['tweet_id']}: {r.json()}")
        except Exception as e:
            print(f"Failed tweet {t['tweet_id']}: {e}")

def seed_documents():
    # Create a minimal PDF in memory (using reportlab)
    try:
        from reportlab.pdfgen import canvas
        from io import BytesIO
        buf = BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 700, "UN OCHA Flood Response Guidelines")
        c.drawString(100, 680, "1. Prioritize search and rescue in submerged areas.")
        c.drawString(100, 660, "2. Establish emergency medical posts within 12 hours.")
        c.drawString(100, 640, "3. Distribute safe drinking water and sanitation kits.")
        c.drawString(100, 620, "4. Activate mental health support for affected populations.")
        c.drawString(100, 600, "5. Coordinate with local authorities for evacuation routes.")
        c.save()
        buf.seek(0)
        files = {"file": ("ocha_guidelines.pdf", buf.read(), "application/pdf")}
        data = {"source": "UN OCHA", "title": "Flood Response Guidelines 2026", "document_type": "guideline"}
        r = requests.post(f"{BASE}/ingest/document", data=data, files=files)
        print(f"Document: {r.json()}")
    except Exception as e:
        print(f"Failed document seed: {e}")

def seed_images():
    try:
        # Create a dummy GeoTIFF-like upload (just a binary placeholder)
        files = {"file": ("demo_sentinel.tif", b"II*\x00\x08\x00\x00\x00" + b"\x00"*100, "image/tiff")}
        data = {
            "source": "Sentinel-2",
            "capture_time": "2026-05-28T12:00:00Z",
            "area": json.dumps({"type":"Polygon","coordinates":[[[-115.5,35.9],[-115.1,35.9],[-115.1,36.3],[-115.5,36.3],[-115.5,35.9]]]}),
            "pre_event": "false",
        }
        r = requests.post(f"{BASE}/ingest/image", data=data, files=files)
        print(f"Image job: {r.json()}")
    except Exception as e:
        print(f"Failed image seed: {e}")

if __name__ == "__main__":
    print("Seeding demo data...")
    seed_tweets()
    seed_documents()
    seed_images()
    print("Done. Check dashboard at http://localhost:3000")
