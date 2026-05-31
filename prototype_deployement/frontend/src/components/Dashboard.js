import React, { useEffect, useState } from 'react';
import { getMetrics, getImageJobs } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const COLORS = ['#48bb78', '#ecc94b', '#ed8936', '#f56565'];

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [mRes, jRes] = await Promise.all([getMetrics(), getImageJobs(5)]);
        setMetrics(mRes.data);
        setJobs(jRes.data || []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="card">Loading dashboard...</div>;
  if (!metrics) return <div className="card">Failed to load metrics.</div>;

  const damageData = Object.entries(metrics.damage_summary || {}).map(([k, v]) => ({ name: k.replace('_', ' '), value: v }));
  const sentimentData = metrics.sentiment_timeline || [];

  return (
    <div>
      <h1 style={{ color: '#1a365d', marginBottom: 20 }}>Dashboard</h1>

      <div className="grid-4" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 20, marginBottom: 20 }}>
        <div className="metric-card">
          <div className="metric-value">{metrics.total_images_processed}</div>
          <div className="metric-label">Images Processed</div>
        </div>
        <div className="metric-card" style={{ background: 'linear-gradient(135deg, #fef5f5 0%, #fed7d7 100%)' }}>
          <div className="metric-value" style={{ color: '#c53030' }}>{metrics.total_tweets_analyzed}</div>
          <div className="metric-label">Tweets Analyzed</div>
        </div>
        <div className="metric-card" style={{ background: 'linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%)' }}>
          <div className="metric-value" style={{ color: '#276749' }}>{metrics.total_documents_indexed}</div>
          <div className="metric-label">Documents Indexed</div>
        </div>
        <div className="metric-card" style={{ background: 'linear-gradient(135deg, #faf5ff 0%, #e9d8fd 100%)' }}>
          <div className="metric-value" style={{ color: '#553c9a' }}>{jobs.filter(j => j.status === 'completed').length}</div>
          <div className="metric-label">Completed Jobs</div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Damage Severity Distribution</h2>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={damageData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label>
                {damageData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h2>Sentiment Timeline (24h)</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={sentimentData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="negative" fill="#f56565" />
              <Bar dataKey="positive" fill="#48bb78" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h2>Recent Alerts</h2>
        {(metrics.recent_alerts || []).map((alert, i) => (
          <div key={i} className="alert-item">⚠️ {alert}</div>
        ))}
      </div>

      <div className="card">
        <h2>Recent Image Jobs</h2>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>
              <th>Job ID</th><th>Source</th><th>Status</th><th>Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map(j => (
              <tr key={j.job_id} style={{ borderBottom: '1px solid #edf2f7' }}>
                <td>{j.job_id.slice(0, 8)}...</td>
                <td>{j.source}</td>
                <td><span style={{
                  padding: '2px 8px', borderRadius: 12, fontSize: '0.8rem',
                  background: j.status === 'completed' ? '#c6f6d5' : j.status === 'failed' ? '#fed7d7' : '#bee3f8',
                  color: j.status === 'completed' ? '#276749' : j.status === 'failed' ? '#c53030' : '#2c5282'
                }}>{j.status}</span></td>
                <td>{new Date(j.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Dashboard;
