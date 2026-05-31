import json
from typing import Dict, Any, List
from shapely.geometry import shape, mapping

from app.core.logging import get_logger

logger = get_logger("services.gis")

class GISExporter:
    @staticmethod
    def to_geojson(features: List[Dict[str, Any]]) -> str:
        """Export features to GeoJSON string."""
        fc = {"type": "FeatureCollection", "features": features}
        return json.dumps(fc, indent=2)

    @staticmethod
    def to_kml(features: List[Dict[str, Any]], name: str = "Disaster Export") -> str:
        """Export features to KML string."""
        kml = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<kml xmlns="http://www.opengis.net/kml/2.2">',
               "<Document>",
               f"<name>{name}</name>"]

        colors = {"no_damage": "ff00ff00", "minor": "ff00ffff", "major": "ff00aaff", "destroyed": "ff0000ff"}

        for feat in features:
            props = feat.get("properties", {})
            sev = props.get("severity", "unknown")
            kml.append("<Placemark>")
            kml.append(f"<name>{sev}</name>")
            kml.append(f"<description>Confidence: {props.get('confidence', 'N/A')}</description>")
            kml.append(f"<Style><LineStyle><color>{colors.get(sev, 'ffffffff')}</color></LineStyle></Style>")
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                kml.append("<Polygon><outerBoundaryIs><LinearRing><coordinates>")
                coords = geom["coordinates"][0]
                kml.append(" ".join([f"{c[0]},{c[1]},0" for c in coords]))
                kml.append("</coordinates></LinearRing></outerBoundaryIs></Polygon>")
            kml.append("</Placemark>")

        kml.extend(["</Document>", "</kml>"])
        return "\n".join(kml)

# Singleton
_gis_exporter = None

def get_gis_exporter() -> GISExporter:
    global _gis_exporter
    if _gis_exporter is None:
        _gis_exporter = GISExporter()
    return _gis_exporter
