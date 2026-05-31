import React, { useState } from 'react';
import { ingestImage, ingestDocument, ingestTweet } from '../services/api';

function DataIngest() {
  const [activeTab, setActiveTab] = useState('image');
  const [message, setMessage] = useState('');

  const [imageForm, setImageForm] = useState({ source: 'Sentinel-2', capture_time: '', file: null, pre_event: false });
  const [docForm, setDocForm] = useState({ source: 'UN OCHA', title: '', document_type: 'guideline', file: null });
  const [tweetForm, setTweetForm] = useState({ tweet_id: '', text: '', user_location: '', timestamp: '' });

  const handleImage = async (e) => {
    e.preventDefault();
    const fd = new FormData();
    fd.append('source', imageForm.source);
    fd.append('capture_time', new Date(imageForm.capture_time).toISOString());
    fd.append('area', JSON.stringify({ type: 'Polygon', coordinates: [[[-115.5, 35.9], [-115.1, 35.9], [-115.1, 36.3], [-115.5, 36.3], [-115.5, 35.9]]] }));
    fd.append('pre_event', imageForm.pre_event);
    fd.append('file', imageForm.file);
    try {
      const res = await ingestImage(fd);
      setMessage(`Image job queued: ${res.data.job_id}`);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleDoc = async (e) => {
    e.preventDefault();
    const fd = new FormData();
    fd.append('source', docForm.source);
    fd.append('title', docForm.title);
    fd.append('document_type', docForm.document_type);
    fd.append('file', docForm.file);
    try {
      const res = await ingestDocument(fd);
      setMessage(`Document indexed: ${res.data.document_id}`);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleTweet = async (e) => {
    e.preventDefault();
    try {
      const res = await ingestTweet({
        tweet_id: tweetForm.tweet_id,
        text: tweetForm.text,
        user_location: tweetForm.user_location || null,
        timestamp: new Date(tweetForm.timestamp).toISOString(),
        language: 'en',
      });
      setMessage(`Tweet ingested: ${res.data.record_id}`);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div>
      <h1 style={{ color: '#1a365d', marginBottom: 20 }}>Data Ingestion</h1>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {['image', 'document', 'tweet'].map(t => (
          <button key={t} className={activeTab === t ? 'btn' : 'btn btn-secondary'} onClick={() => { setActiveTab(t); setMessage(''); }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {message && <div className="card" style={{ background: '#f0fff4', borderLeft: '4px solid #48bb78' }}>{message}</div>}

      {activeTab === 'image' && (
        <div className="card">
          <h2>Ingest Satellite Imagery</h2>
          <form onSubmit={handleImage}>
            <div className="form-group"><label>Source</label><input value={imageForm.source} onChange={e => setImageForm({ ...imageForm, source: e.target.value })} /></div>
            <div className="form-group"><label>Capture Time</label><input type="datetime-local" onChange={e => setImageForm({ ...imageForm, capture_time: e.target.value })} required /></div>
            <div className="form-group"><label>GeoTIFF File</label><input type="file" accept=".tif,.tiff" onChange={e => setImageForm({ ...imageForm, file: e.target.files[0] })} required /></div>
            <label><input type="checkbox" checked={imageForm.pre_event} onChange={e => setImageForm({ ...imageForm, pre_event: e.target.checked })} /> Pre-event image</label>
            <div style={{ marginTop: 12 }}><button className="btn" type="submit">Upload & Process</button></div>
          </form>
        </div>
      )}

      {activeTab === 'document' && (
        <div className="card">
          <h2>Ingest PDF Document</h2>
          <form onSubmit={handleDoc}>
            <div className="form-group"><label>Source Organization</label><input value={docForm.source} onChange={e => setDocForm({ ...docForm, source: e.target.value })} /></div>
            <div className="form-group"><label>Title</label><input value={docForm.title} onChange={e => setDocForm({ ...docForm, title: e.target.value })} required /></div>
            <div className="form-group"><label>Type</label>
              <select value={docForm.document_type} onChange={e => setDocForm({ ...docForm, document_type: e.target.value })}>
                <option value="guideline">Guideline</option>
                <option value="situation_report">Situation Report</option>
                <option value="technical_manual">Technical Manual</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="form-group"><label>PDF File</label><input type="file" accept=".pdf" onChange={e => setDocForm({ ...docForm, file: e.target.files[0] })} required /></div>
            <button className="btn" type="submit">Index Document</button>
          </form>
        </div>
      )}

      {activeTab === 'tweet' && (
        <div className="card">
          <h2>Ingest Social Media Post</h2>
          <form onSubmit={handleTweet}>
            <div className="form-group"><label>Tweet ID</label><input value={tweetForm.tweet_id} onChange={e => setTweetForm({ ...tweetForm, tweet_id: e.target.value })} required /></div>
            <div className="form-group"><label>Text</label><textarea rows={3} value={tweetForm.text} onChange={e => setTweetForm({ ...tweetForm, text: e.target.value })} required /></div>
            <div className="form-group"><label>User Location (optional)</label><input value={tweetForm.user_location} onChange={e => setTweetForm({ ...tweetForm, user_location: e.target.value })} /></div>
            <div className="form-group"><label>Timestamp</label><input type="datetime-local" onChange={e => setTweetForm({ ...tweetForm, timestamp: e.target.value })} required /></div>
            <button className="btn" type="submit">Ingest Tweet</button>
          </form>
        </div>
      )}
    </div>
  );
}

export default DataIngest;
