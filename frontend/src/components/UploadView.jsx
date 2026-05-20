import { useState, useRef, useEffect } from 'react';
import { Upload, X, Play, Download, Check, AlertTriangle, FileText, Image as ImageIcon } from 'lucide-react';
import { useToast } from './Toast';

export default function UploadView() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [labRoom, setLabRoom] = useState('Lab B01');
  const [uploadedBy, setUploadedBy] = useState('Daa(HIT)');
  const [customUploadedBy, setCustomUploadedBy] = useState('');
  const [status, setStatus] = useState('idle'); // idle, uploading, analyzing, complete, error
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [activeImageTab, setActiveImageTab] = useState('annotated');
  
  const fileInputRef = useRef(null);
  const toast = useToast();

  const parseJsonSafely = async (response) => {
    const text = await response.text();
    try {
      return JSON.parse(text);
    } catch (error) {
      if (text.trim().startsWith('<')) {
        throw new Error(`Server returned HTML (Backend might not be running or crashed): ${response.status} ${response.statusText}`);
      }
      throw new Error(`Invalid server response (${response.status}): ${text.slice(0, 200)}`);
    }
  };

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (selectedFile) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(selectedFile.type)) {
      toast('Please upload a JPG or PNG image.', 'error');
      return;
    }
    if (selectedFile.size > 50 * 1024 * 1024) {
      toast('File too large. Max 50MB.', 'error');
      return;
    }
    setFile(selectedFile);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(selectedFile);
    setResult(null);
    setStatus('idle');
  };

  const clearFile = () => {
    setFile(null); setPreview(null); setResult(null); setStatus('idle');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const animateProgress = (from, to, duration) => {
    const start = Date.now();
    return new Promise(resolve => {
      const tick = () => {
        const elapsed = Date.now() - start;
        const pct = Math.min(from + (to - from) * (elapsed / duration), to);
        setProgress(pct);
        if (elapsed < duration) requestAnimationFrame(tick);
        else resolve();
      };
      tick();
    });
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus('uploading');
    setProgress(0);

    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('lab_room', labRoom);
      const finalUploadedBy = uploadedBy === 'Other' ? customUploadedBy : uploadedBy;
      fd.append('uploaded_by', finalUploadedBy || 'Admin');

      await animateProgress(0, 40, 1500);
      const uploadRes = await fetch('/api/upload', { method: 'POST', body: fd });
      const uploadJson = await parseJsonSafely(uploadRes);
      
      if (!uploadRes.ok || !uploadJson.success) throw new Error(uploadJson.detail || 'Upload failed');

      const analysisId = uploadJson.data.id;
      setStatus('analyzing');
      await animateProgress(40, 90, 5000);

      const analyzeRes = await fetch(`/api/analyze/${analysisId}`, { method: 'POST' });
      const analyzeJson = await parseJsonSafely(analyzeRes);
      
      if (!analyzeRes.ok || !analyzeJson.success) throw new Error(analyzeJson.detail || 'Analysis failed');

      setProgress(100);
      setResult(analyzeJson.data);
      setStatus('complete');
      toast('Analysis completed successfully!', 'success');

      setTimeout(() => {
        document.getElementById('result-card')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 300);

    } catch (err) {
      setStatus('error');
      toast(`Error: ${err.message}`, 'error');
    }
  };

  return (
    <div className="upload-section animate-fade-in">
      <div className="section-header">
        <h1 className="gradient-text-animate">Chair Arrangement Detection</h1>
        <p>Upload a lab or classroom photo to instantly detect chair arrangements using a local ML model</p>
      </div>

      <div className="upload-grid">
        {/* Upload Card */}
        <div className="card upload-card">
          <div className="card-header">
            <h2>📷 Upload Image</h2>
          </div>
          
          <div 
            className={`dropzone ${isDragging ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !preview && fileInputRef.current?.click()}
          >
            {preview ? (
              <img src={preview} className="preview-image" alt="Preview" />
            ) : (
              <div className="dropzone-content">
                <Upload size={48} className="dropzone-icon" />
                <p className="dropzone-text">Drag & drop your image here</p>
                <p className="dropzone-subtext">or click to browse files</p>
                <span className="dropzone-formats">Supports: JPG, JPEG, PNG (max 50MB)</span>
              </div>
            )}
            <input 
              type="file" 
              ref={fileInputRef} 
              accept=".jpg,.jpeg,.png" 
              className="hidden" 
              onChange={handleFileSelect} 
            />
          </div>

          <div className="upload-options">
            <div className="form-group">
              <label>Lab Room</label>
              <select value={labRoom} onChange={e => setLabRoom(e.target.value)} disabled={status === 'uploading' || status === 'analyzing'}>
                <option value="Lab B01">Lab B01</option>
                <option value="Lab B02">Lab B02</option>
                <option value="Lab B03">Lab B03</option>
                <option value="Lab B04">Lab B04</option>
                <option value="Lab B05">Lab B05</option>
                <option value="Lab B06">Lab B06</option>
                <option value="Lab B07">Lab B07</option>
                <option value="Lab B08">Lab B08</option>
                <option value="Lab B09">Lab B09</option>
                <option value="Lab B10">Lab B10</option>
                <option value="Lab B11">Lab B11</option>
              </select>
            </div>
            <div className="form-group">
              <label>Uploaded By</label>
              <select value={uploadedBy} onChange={e => setUploadedBy(e.target.value)} disabled={status === 'uploading' || status === 'analyzing'}>
                <option value="Daa(HIT)">Daa(HIT)</option>
                <option value="Oops(HIT)">Oops(HIT)</option>
                <option value="OS(HIT)">OS(HIT)</option>
                <option value="Other">Other</option>
              </select>
              {uploadedBy === 'Other' && (
                <input 
                  type="text" 
                  value={customUploadedBy} 
                  onChange={e => setCustomUploadedBy(e.target.value)} 
                  placeholder="Enter your name"
                  disabled={status === 'uploading' || status === 'analyzing'}
                  style={{ marginTop: '8px', width: '100%' }}
                />
              )}
            </div>
          </div>

          <div className="upload-actions">
            <button className="btn btn-secondary" onClick={clearFile} disabled={!file || status === 'uploading' || status === 'analyzing'}>
              <X size={16} /> Clear
            </button>
            <button className="btn btn-primary" onClick={handleAnalyze} disabled={!file || status === 'uploading' || status === 'analyzing'}>
              <Play size={16} fill="currentColor" /> Analyze with ML Model
            </button>
          </div>

          {(status === 'uploading' || status === 'analyzing') && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
              </div>
              <p className="progress-text">
                {status === 'uploading' ? 'Uploading image...' : 'Running ML model analysis...'} {Math.round(progress)}%
              </p>
            </div>
          )}
        </div>

        {/* Result Card */}
        {result && (
          <div className="card result-card" id="result-card">
            <div className="card-header">
              <h2>🔍 Analysis Results</h2>
              <div className={`status-badge ${result.misplaced_chairs === 0 && result.total_chairs > 0 ? 'success' : result.misplaced_chairs > 0 ? 'danger' : 'warning'}`}>
                {result.misplaced_chairs === 0 && result.total_chairs > 0 ? 'All Clear' : result.misplaced_chairs > 0 ? 'Issues Found' : 'No Chairs'}
              </div>
            </div>

            <div className="result-summary">
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-value">{result.total_chairs}</span>
                  <span className="stat-label">Total Chairs</span>
                </div>
                <div className="stat-item stat-success">
                  <span className="stat-value">{result.correct_chairs}</span>
                  <span className="stat-label">Properly Arranged</span>
                </div>
                <div className="stat-item stat-danger">
                  <span className="stat-value">{result.misplaced_chairs}</span>
                  <span className="stat-label">Misplaced</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{result.accuracy}%</span>
                  <span className="stat-label">Accuracy</span>
                </div>
              </div>

              <div className="confidence-section">
                <div className="confidence-header">
                  <span>ML Confidence</span>
                  <span className="confidence-value">{result.avg_confidence}%</span>
                </div>
                <div className="confidence-bar">
                  <div className="confidence-fill" style={{ width: `${result.avg_confidence}%` }}></div>
                </div>
              </div>
            </div>

            <div className="image-tabs">
              <button className={`image-tab ${activeImageTab === 'annotated' ? 'active' : ''}`} onClick={() => setActiveImageTab('annotated')}>Annotated</button>
              <button className={`image-tab ${activeImageTab === 'original' ? 'active' : ''}`} onClick={() => setActiveImageTab('original')}>Original</button>
            </div>
            
            <div className="result-image-container">
              {activeImageTab === 'annotated' && result.result_image_url && <img src={result.result_image_url} alt="Annotated" className="result-image" />}
              {activeImageTab === 'original' && result.upload_image_url && <img src={result.upload_image_url} alt="Original" className="result-image" />}
            </div>

            {result.ai_description && (
              <div className="chair-details" style={{ marginTop: '16px' }}>
                <h3>🤖 ML Model Summary</h3>
                <div style={{ padding: '14px 16px', background: 'var(--bg-main)', borderRadius: '8px', lineHeight: '1.6', color: 'var(--text-primary)', fontSize: '14px' }}>
                  {result.ai_description}
                </div>
              </div>
            )}

            <div className="chair-details">
              <h3>🚨 Misplaced Chairs Breakdown</h3>
              <div className="chair-list">
                {(() => {
                  const misplaced = (result.details?.chairs || []).filter(c => !c.is_properly_arranged);
                  if (misplaced.length > 0) {
                    return misplaced.map((c, i) => (
                      <div key={i} className="chair-item misplaced">
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

            <div className="result-actions">
              {result.pdf_report_url && (
                <a href={result.pdf_report_url} target="_blank" rel="noreferrer" className="btn btn-primary" style={{ textDecoration: 'none' }}>
                  <FileText size={16} /> Download PDF Report
                </a>
              )}
              {result.result_image_url && (
                <a href={result.result_image_url} download="analysis_result.jpg" className="btn btn-secondary" style={{ textDecoration: 'none' }}>
                  <ImageIcon size={16} /> Download Image
                </a>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
