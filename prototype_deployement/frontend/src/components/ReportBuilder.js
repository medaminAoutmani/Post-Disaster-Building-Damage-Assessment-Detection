import React, { useState } from 'react';
import { generateReport, getReport, ragQuery } from '../services/api';

function ReportBuilder() {
  const [form, setForm] = useState({
    title: '',
    start_date: '',
    end_date: '',
    include_damage: true,
    include_sentiment: true,
    include_rag: true,
    format: 'pdf',
  });
  const [reportId, setReportId] = useState(null);
  const [status, setStatus] = useState('');
  const [ragResult, setRagResult] = useState(null);
  const [ragQueryText, setRagQueryText] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus('Submitting...');
    try {
      const payload = {
        title: form.title,
        region: { type: 'Polygon', coordinates: [[[-115.5, 35.9], [-115.1, 35.9], [-115.1, 36.3], [-115.5, 36.3], [-115.5, 35.9]]] },
        start_date: new Date(form.start_date).toISOString(),
        end_date: new Date(form.end_date).toISOString(),
        include_damage: form.include_damage,
        include_sentiment: form.include_sentiment,
        include_rag: form.include_rag,
        format: form.format,
      };
      const res = await generateReport(payload);
      setReportId(res.data.report_id);
      setStatus(`Report queued: ${res.data.report_id}`);
      pollStatus(res.data.report_id);
    } catch (err) {
      setStatus('Error: ' + (err.response?.data?.detail || err.message));
    }
  };

  const pollStatus = async (id) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const res = await getReport(id);
        setStatus(`Status: ${res.data.status}`);
        if (res.data.status === 'completed' || res.data.status === 'failed' || attempts > 30) {
          clearInterval(interval);
        }
      } catch (e) {
        clearInterval(interval);
      }
    }, 2000);
  };

  const handleRag = async () => {
    if (!ragQueryText) return;
    setRagResult(null);
    try {
      const res = await ragQuery({
        query: ragQueryText,
        damage_context: { severity: 'major', affected_area: 'sector_7' },
        top_k: 5,
      });
      setRagResult(res.data);
    } catch (err) {
      setRagResult({ commentary: 'Error: ' + (err.response?.data?.detail || err.message) });
    }
  };

  return (
    <div>
      <h1 style={{ color: '#1a365d', marginBottom: 20 }}>Report Builder & RAG Analysis</h1>

      <div className="grid-2">
        <div className="card">
          <h2>Generate Report</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Report Title</label>
              <input value={form.title} onChange={e => setForm({ ...form, title: e.target.value })} required />
            </div>
            <div className="form-group">
              <label>Start Date</label>
              <input type="datetime-local" value={form.start_date} onChange={e => setForm({ ...form, start_date: e.target.value })} required />
            </div>
            <div className="form-group">
              <label>End Date</label>
              <input type="datetime-local" value={form.end_date} onChange={e => setForm({ ...form, end_date: e.target.value })} required />
            </div>
            <div className="form-group">
              <label>Format</label>
              <select value={form.format} onChange={e => setForm({ ...form, format: e.target.value })}>
                <option value="pdf">PDF</option>
                <option value="html">HTML</option>
                <option value="json">JSON</option>
                <option value="geojson">GeoJSON</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
              <label><input type="checkbox" checked={form.include_damage} onChange={e => setForm({ ...form, include_damage: e.target.checked })} /> Damage</label>
              <label><input type="checkbox" checked={form.include_sentiment} onChange={e => setForm({ ...form, include_sentiment: e.target.checked })} /> Sentiment</label>
              <label><input type="checkbox" checked={form.include_rag} onChange={e => setForm({ ...form, include_rag: e.target.checked })} /> RAG</label>
            </div>
            <button className="btn" type="submit">Generate Report</button>
          </form>
          {status && <p style={{ marginTop: 12, color: '#2c5282' }}>{status}</p>}
          {reportId && <p style={{ fontSize: '0.85rem', color: '#718096' }}>Report ID: {reportId}</p>}
        </div>

        <div className="card">
          <h2>RAG Expert Query</h2>
          <div className="form-group">
            <label>Ask the Knowledge Base</label>
            <textarea rows={4} value={ragQueryText} onChange={e => setRagQueryText(e.target.value)} placeholder="e.g., What are the priority actions for flood response?" />
          </div>
          <button className="btn" onClick={handleRag}>Query</button>
          {ragResult && (
            <div style={{ marginTop: 16, padding: 12, background: '#f7fafc', borderRadius: 6, fontSize: '0.9rem' }}>
              <p><b>Commentary:</b></p>
              <p style={{ whiteSpace: 'pre-wrap' }}>{ragResult.commentary}</p>
              {ragResult.citations?.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <b>Citations:</b>
                  <ul>{ragResult.citations.map((c, i) => <li key={i}>{c}</li>)}</ul>
                </div>
              )}
              <p style={{ marginTop: 8, fontSize: '0.8rem', color: '#718096' }}>Confidence: {ragResult.confidence}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ReportBuilder;
