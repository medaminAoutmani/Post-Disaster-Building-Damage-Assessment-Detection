import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: `${API_BASE}/`,
  headers: { 'Content-Type': 'application/json' },
});

export const getMetrics = () => api.get('/dashboard/metrics');
export const getImageJobs = (limit = 20) => api.get(`/jobs/image?limit=${limit}`);
export const getImageJob = (id) => api.get(`/jobs/image/${id}`);
export const ingestImage = (formData) => api.post('/ingest/image', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
export const ingestDocument = (formData) => api.post('/ingest/document', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
export const ingestTweet = (payload) => api.post('/ingest/tweet', payload);
export const generateReport = (payload) => api.post('/reports/generate', payload);
export const getReport = (id) => api.get(`/reports/${id}`);
export const ragQuery = (payload) => api.post('/rag/query', payload);

export default api;
