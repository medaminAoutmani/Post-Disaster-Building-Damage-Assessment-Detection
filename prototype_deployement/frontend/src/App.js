import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import MapViewer from './components/MapViewer';
import ReportBuilder from './components/ReportBuilder';
import DataIngest from './components/DataIngest';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">🌍 PDA Platform</div>
          <ul className="nav-links">
            <li><NavLink to="/" end>Dashboard</NavLink></li>
            <li><NavLink to="/map">Map Viewer</NavLink></li>
            <li><NavLink to="/reports">Reports</NavLink></li>
            <li><NavLink to="/ingest">Data Ingest</NavLink></li>
          </ul>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/map" element={<MapViewer />} />
            <Route path="/reports" element={<ReportBuilder />} />
            <Route path="/ingest" element={<DataIngest />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
