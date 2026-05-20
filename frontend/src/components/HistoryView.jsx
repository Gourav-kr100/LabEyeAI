import { useState, useEffect, useCallback } from 'react';
import { Filter, FileText, Trash2, Eye, X, AlertTriangle, Check, Image as ImageIcon } from 'lucide-react';
import { useToast } from './Toast';

export default function HistoryView() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({ search: '', status: '', lab_room: '', date_from: '' });
  const [selectedItem, setSelectedItem] = useState(null);
  const [activeImageTab, setActiveImageTab] = useState('annotated');
  const toast = useToast();

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filters.search) params.set('search', filters.search);
      if (filters.status) params.set('status', filters.status);
      if (filters.lab_room) params.set('lab_room', filters.lab_room);
      if (filters.date_from) params.set('date_from', filters.date_from);

      const res = await fetch(`/api/history?${params.toString()}`);
      const json = await res.json();
      if (json.success) {
        setHistory(json.data);
      }
    } catch (err) {
      console.error('History load error:', err);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleFilterChange = (e) => {
    setFilters(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleDelete = async (analysisId) => {
    if (!window.confirm('Delete this history entry? This cannot be undone.')) {
      return;
    }
    try {
      const res = await fetch(`/api/history/${analysisId}`, { method: 'DELETE' });
      const json = await res.json();
      if (!res.ok || !json.success) {
        throw new Error(json.detail || 'Delete failed');
      }
      setHistory(prev => prev.filter(item => item.id !== analysisId));
      toast('History entry deleted successfully.', 'success');
    } catch (err) {
      toast(`Unable to delete history: ${err.message}`, 'error');
    }
  };

  const handleClearAll = async () => {
    if (!history.length) {
      toast('No history entries to clear.', 'info');
      return;
    }
    if (!window.confirm('Delete all history entries? This action cannot be undone.')) {
      return;
    }
    try {
      const res = await fetch('/api/history', { method: 'DELETE' });
      const json = await res.json();
      if (!res.ok || !json.success) {
        throw new Error(json.detail || 'Clear failed');
      }
      setHistory([]);
      toast('All history entries deleted.', 'success');
    } catch (err) {
      toast(`Unable to clear history: ${err.message}`, 'error');
    }
  };

  return (
    <div className="history-section animate-fade-in">
      <div className="section-header">
        <h1>Analysis History</h1>
        <p>Browse all previous chair arrangement analyses</p>
      </div>

      <div className="card filter-card">
        <div className="filter-grid">
          <div className="form-group">
            <label>Search</label>
            <input name="search" type="text" placeholder="Search by filename..." value={filters.search} onChange={handleFilterChange} />
          </div>
          <div className="form-group">
            <label>Status</label>
            <select name="status" value={filters.status} onChange={handleFilterChange}>
              <option value="">All</option>
              <option value="completed">Completed</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          <div className="form-group">
            <label>Lab Room</label>
            <select name="lab_room" value={filters.lab_room} onChange={handleFilterChange}>
              <option value="">All Labs</option>
              <option value="Lab 1">Lab 1</option>
              <option value="Lab 2">Lab 2</option>
              <option value="Lab 3">Lab 3</option>
              <option value="Computer Lab">Computer Lab</option>
              <option value="Physics Lab">Physics Lab</option>
              <option value="Chemistry Lab">Chemistry Lab</option>
            </select>
          </div>
          <div className="form-group">
            <label>Date From</label>
            <input name="date_from" type="date" value={filters.date_from} onChange={handleFilterChange} />
          </div>
          <div className="filter-actions">
            <button className="btn btn-primary filter-btn" onClick={fetchHistory}>
              <Filter size={16} /> Apply Filters
            </button>
            <button className="btn btn-danger filter-btn" onClick={handleClearAll} type="button">
              <Trash2 size={16} /> Clear All History
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Image</th>
                <th>Lab Room</th>
                <th>Chairs</th>
                <th>Accuracy</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="7" className="empty-state">Loading...</td></tr>
              ) : history.length > 0 ? (
                history.map((a, i) => {
                  const date = new Date(a.created_at).toLocaleString();
                  const accCls = a.accuracy >= 80 ? 'success' : a.accuracy >= 50 ? 'warning' : 'danger';
                  const statusCls = a.status === 'completed' ? 'success' : a.status === 'failed' ? 'danger' : 'pending';
                  
                  return (
                    <tr key={i}>
                      <td>{date}</td>
                      <td>{a.upload_image_url ? <img src={a.upload_image_url} className="history-thumb" alt="thumb" /> : '—'}</td>
                      <td>{a.lab_room}</td>
                      <td>{a.total_chairs} ({a.misplaced_chairs} ⚠️)</td>
                      <td><span className={`status-badge ${accCls}`}>{a.accuracy}%</span></td>
                      <td><span className={`status-badge ${statusCls}`}>{a.status}</span></td>
                      <td>
                        <div className="history-actions">
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              setSelectedItem(a);
                              setActiveImageTab('annotated');
                            }}
                            type="button"
                            style={{ padding: '4px 10px', fontSize: '12px' }}
                          >
                            <Eye size={12} /> View
                          </button>
                          {a.pdf_report_url ? (
                            <a href={a.pdf_report_url} target="_blank" rel="noreferrer" className="btn btn-secondary btn-sm" style={{ padding: '4px 10px', fontSize: '12px', textDecoration: 'none' }}>
                              <FileText size={12} /> PDF
                            </a>
                          ) : '—'}
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={() => handleDelete(a.id)}
                            type="button"
                            style={{ padding: '4px 10px', fontSize: '12px' }}
                          >
                            <Trash2 size={12} /> Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })
              ) : (
                <tr><td colSpan="7" className="empty-state">No analyses found. Upload an image to get started!</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedItem && (
        <div className="modal-overlay" onClick={() => setSelectedItem(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="card-header" style={{ padding: '16px 24px' }}>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: '700' }}>
                  🔍 Analysis Details
                </h2>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  {selectedItem.original_filename} — {new Date(selectedItem.created_at).toLocaleString()}
                </p>
              </div>
              <button 
                className="theme-toggle" 
                onClick={() => setSelectedItem(null)}
                style={{ width: '32px', height: '32px' }}
              >
                <X size={16} />
              </button>
            </div>

            <div className="modal-body">
              <div className="result-summary" style={{ paddingTop: '16px' }}>
                <div className="stat-grid" style={{ padding: '0 24px 16px' }}>
                  <div className="stat-item">
                    <span className="stat-value">{selectedItem.total_chairs}</span>
                    <span className="stat-label">Total Chairs</span>
                  </div>
                  <div className="stat-item stat-success">
                    <span className="stat-value">{selectedItem.correct_chairs}</span>
                    <span className="stat-label">Properly Arranged</span>
                  </div>
                  <div className="stat-item stat-danger">
                    <span className="stat-value">{selectedItem.misplaced_chairs}</span>
                    <span className="stat-label">Misplaced</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{selectedItem.accuracy}%</span>
                    <span className="stat-label">Accuracy</span>
                  </div>
                </div>

                <div className="confidence-section">
                  <div className="confidence-header">
                    <span>ML Confidence</span>
                    <span className="confidence-value">{selectedItem.avg_confidence}%</span>
                  </div>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: `${selectedItem.avg_confidence}%` }}></div>
                  </div>
                </div>
              </div>

              <div className="image-tabs" style={{ marginTop: '8px' }}>
                <button className={`image-tab ${activeImageTab === 'annotated' ? 'active' : ''}`} onClick={() => setActiveImageTab('annotated')}>Annotated</button>
                <button className={`image-tab ${activeImageTab === 'original' ? 'active' : ''}`} onClick={() => setActiveImageTab('original')}>Original</button>
              </div>
              
              <div className="result-image-container" style={{ margin: '0 24px 16px' }}>
                {activeImageTab === 'annotated' && selectedItem.result_image_url && (
                  <img src={selectedItem.result_image_url} alt="Annotated" className="result-image" />
                )}
                {activeImageTab === 'original' && selectedItem.upload_image_url && (
                  <img src={selectedItem.upload_image_url} alt="Original" className="result-image" />
                )}
                {((activeImageTab === 'annotated' && !selectedItem.result_image_url) || 
                  (activeImageTab === 'original' && !selectedItem.upload_image_url)) && (
                  <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Image not available</div>
                )}
              </div>

              {selectedItem.ai_description && (
                <div className="chair-details" style={{ paddingTop: '0', paddingBottom: '16px' }}>
                  <h3>🤖 ML Model Summary</h3>
                  <div style={{ padding: '14px 16px', background: 'var(--bg-input)', borderRadius: '8px', lineHeight: '1.6', color: 'var(--text-primary)', fontSize: '14px' }}>
                    {selectedItem.ai_description}
                  </div>
                </div>
              )}

              <div className="chair-details" style={{ paddingTop: '0' }}>
                <h3>🚨 Misplaced Chairs Breakdown</h3>
                <div className="chair-list">
                  {(() => {
                    const misplaced = (selectedItem.details?.chairs || []).filter(c => !c.is_properly_arranged);
                    if (misplaced.length > 0) {
                      return misplaced.map((c, idx) => (
                        <div key={idx} className="chair-item misplaced">
                          <span className="chair-icon"><AlertTriangle size={20} color="var(--danger)" /></span>
                          <div className="chair-info">
                            <div className="chair-name">Chair #{c.chair_id}</div>
                            <div className="chair-status">{(c.issues || []).join(', ') || 'Misplaced'}</div>
                          </div>
                          <span className="chair-score bad">
                            {Math.round(c.alignment_score)}%
                          </span>
                        </div>
                      ));
                    } else {
                      return (
                        <div className="chair-item" style={{ borderLeft: '4px solid var(--success)', padding: '15px' }}>
                          <span className="chair-icon"><Check size={20} color="var(--success)" /></span>
                          <div className="chair-info">
                            <div className="chair-name" style={{ color: 'var(--success)', fontWeight: 'bold' }}>All Clear!</div>
                            <div className="chair-status">Every chair is properly tucked in and positioned.</div>
                          </div>
                        </div>
                      );
                    }
                  })()}
                </div>
              </div>
            </div>

            <div className="modal-footer">
              {selectedItem.pdf_report_url && (
                <a href={selectedItem.pdf_report_url} target="_blank" rel="noreferrer" className="btn btn-primary btn-sm" style={{ textDecoration: 'none' }}>
                  <FileText size={14} /> PDF Report
                </a>
              )}
              {selectedItem.result_image_url && (
                <a href={selectedItem.result_image_url} download="analysis_result.jpg" className="btn btn-secondary btn-sm" style={{ textDecoration: 'none' }}>
                  <ImageIcon size={14} /> Download Image
                </a>
              )}
              <button className="btn btn-secondary btn-sm" onClick={() => setSelectedItem(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
