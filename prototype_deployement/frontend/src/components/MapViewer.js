import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON, LayersControl, LayerGroup, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { getImageJobs, getMetrics } from '../services/api';

function MapViewer() {
  const [damageFeatures, setDamageFeatures] = useState([]);
  const [tweetPoints, setTweetPoints] = useState([]);
  const [center] = useState([36.1, -115.2]); // Las Vegas example

  useEffect(() => {
    async function load() {
      try {
        const jRes = await getImageJobs(50);
        const allFeatures = [];
        for (const job of jRes.data || []) {
          if (job.map_url) {
            // In real app, fetch GeoJSON from backend; here mock features
            allFeatures.push({
              type: 'Feature',
              geometry: { type: 'Polygon', coordinates: [[
                [-115.3, 36.0], [-115.1, 36.0], [-115.1, 36.2], [-115.3, 36.2], [-115.3, 36.0]
              ]] },
              properties: { severity: 'major', confidence: 0.89, job_id: job.job_id }
            });
          }
        }
        setDamageFeatures(allFeatures);

        // Mock tweet points
        setTweetPoints([
          { coords: [36.12, -115.18], sentiment: 'negative', emotion: 'fear', text: 'City center submerged!' },
          { coords: [36.08, -115.22], sentiment: 'negative', emotion: 'sadness', text: 'Lost everything in the flood' },
          { coords: [36.15, -115.12], sentiment: 'positive', emotion: 'joy', text: 'Rescue teams arrived!' },
        ]);
      } catch (e) {
        console.error(e);
      }
    }
    load();
  }, []);

  const severityStyle = (feature) => {
    const s = feature.properties.severity;
    const colors = { no_damage: '#48bb78', minor: '#ecc94b', major: '#ed8936', destroyed: '#f56565' };
    return { color: colors[s] || '#999', weight: 2, fillOpacity: 0.4, fillColor: colors[s] || '#999' };
  };

  const sentimentColor = (s) => ({ positive: '#48bb78', negative: '#f56565', neutral: '#a0aec0' }[s] || '#999');

  return (
    <div style={{ height: 'calc(100vh - 104px)' }}>
      <h1 style={{ color: '#1a365d', marginBottom: 12 }}>Map Viewer</h1>
      <MapContainer center={center} zoom={12} style={{ height: '100%', borderRadius: 8 }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <LayersControl position="topright">
          <LayersControl.Overlay checked name="Damage Polygons">
            <LayerGroup>
              <GeoJSON data={{ type: 'FeatureCollection', features: damageFeatures }} style={severityStyle} />
            </LayerGroup>
          </LayersControl.Overlay>
          <LayersControl.Overlay checked name="Social Media Sentiment">
            <LayerGroup>
              {tweetPoints.map((t, i) => (
                <CircleMarker key={i} center={t.coords} radius={8} fillColor={sentimentColor(t.sentiment)} color="#333" weight={1} fillOpacity={0.8}>
                  <Popup>
                    <div style={{ fontSize: '0.85rem' }}>
                      <b>Sentiment:</b> {t.sentiment}<br/>
                      <b>Emotion:</b> {t.emotion}<br/>
                      <b>Text:</b> {t.text}
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </LayerGroup>
          </LayersControl.Overlay>
        </LayersControl>
      </MapContainer>
    </div>
  );
}

export default MapViewer;
