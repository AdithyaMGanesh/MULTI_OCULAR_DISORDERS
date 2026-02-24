import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './AppDualModel.css';

const API_BASE_URL = 'http://localhost:8001';

export default function AppDualModel() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  const processFile = (file) => {
    if (file) {
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
      if (!validTypes.includes(file.type)) {
        setError('❌ Invalid file type. Please select PNG, JPG, BMP, or TIFF');
        setSelectedFile(null);
        setPreview(null);
        return;
      }

      if (file.size > 50 * 1024 * 1024) {
        setError('❌ File too large. Maximum 50MB allowed.');
        return;
      }

      setSelectedFile(file);
      setError(null);

      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('❌ Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API_BASE_URL}/predict-dual`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });

      if (response.data.status === 'success') {
        setResult(response.data);
        window.scrollTo(0, document.body.scrollHeight);
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Unknown error';
      setError(`❌ Analysis failed: ${errorMsg}`);
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    if (!severity) return '#3b82f6';
    if (severity === 'Low Risk') return '#10b981';
    if (severity === 'Medium Risk') return '#f59e0b';
    if (severity === 'High Risk') return '#dc2626';
    return '#3b82f6';
  };

  const getSeverityBg = (severity) => {
    if (!severity) return '#f0f9ff';
    if (severity === 'Low Risk') return '#f0fdf4';
    if (severity === 'Medium Risk') return '#fffbeb';
    if (severity === 'High Risk') return '#fef2f2';
    return '#f0f9ff';
  };

  const reset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    fileInputRef.current.value = '';
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-wrapper">
          <div className="header-left">
            <button
              className="back-button"
              onClick={() => navigate('/')}
              title="Back to Home"
            >
              ← Back to Home
            </button>
            <div className="logo-section">
              <div className="logo-icon">👁️</div>
              <div className="logo-text">
                <h1>Multi-Ocular Disease Detection</h1>
              </div>
            </div>
          </div>
          <div className="badge-group">
            <span className="badge badge-dual">🚗 Dual Lane</span>
            <span className="badge badge-live">● Live</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="section upload-section">
          <div className="section-header">
            <h2>📤 Upload Retinal Image</h2>
            <p>Select a fundus photograph for AI analysis</p>
          </div>

          <div
            className={`upload-zone ${selectedFile ? 'with-preview' : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            {!preview ? (
              <div className="upload-content">
                <div className="upload-icon">📷</div>
                <p className="upload-title">Drag & drop image here</p>
                <p className="upload-hint">or click to browse</p>
              </div>
            ) : (
              <div className="preview-wrapper">
                <img src={preview} alt="Preview" className="preview-image" />
                <div className="file-info">{selectedFile?.name}</div>
              </div>
            )}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept="image/*"
              style={{ display: 'none' }}
            />
          </div>

          {error && (
            <div className="alert alert-error">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <div className="action-buttons">
            <button
              className={`btn btn-primary ${loading || !selectedFile ? 'disabled' : ''}`}
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <span className="loader"></span> Analyzing...
                </>
              ) : (
                <>🔬 Analyze (Dual Model)</>
              )}
            </button>
            {selectedFile && (
              <button className="btn btn-secondary" onClick={reset}>
                🔄 Clear
              </button>
            )}
          </div>

          <div className="format-hint">
            <strong>Formats:</strong> PNG, JPG, BMP, TIFF | <strong>Max:</strong> 50MB
          </div>
        </div>

        {result && (
          <>
            {/* Results Section */}
            <div className="results-wrapper">
              <h2 className="results-title">📊 Analysis Results</h2>

              {/* Lane 1: DR */}
              {result.lane_1_specialist && (
                <div className="result-card lane-1-card">
                  <div className="card-header">
                    <div className="lane-badge">🚗 Lane 1</div>
                    <h3>Diabetic Retinopathy (DR)</h3>
                    <span className="card-icon">👁️</span>
                  </div>

                  <div className="diagnosis-container">
                    <div
                      className="diagnosis-panel"
                      style={{
                        backgroundColor: getSeverityBg(result.lane_1_specialist.severity),
                        borderLeft: `5px solid ${getSeverityColor(result.lane_1_specialist.severity)}`,
                      }}
                    >
                      <div className="diagnosis-section">
                        <label>Diagnosis</label>
                        <div className="diagnosis-value">
                          {result.lane_1_specialist.diagnosis}
                        </div>
                      </div>

                      <div className="confidence-section">
                        <label>Confidence Score</label>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{
                              width: `${result.lane_1_specialist.confidence * 100}%`,
                              backgroundColor: getSeverityColor(result.lane_1_specialist.severity),
                            }}
                          ></div>
                        </div>
                        <div className="confidence-text">
                          {(result.lane_1_specialist.confidence * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="severity-section">
                        <label>Risk Level</label>
                        <div
                          className="severity-badge"
                          style={{
                            backgroundColor: getSeverityColor(result.lane_1_specialist.severity),
                          }}
                        >
                          {result.lane_1_specialist.severity}
                        </div>
                      </div>
                    </div>

                    {result.lane_1_specialist.probabilities && (
                      <div className="probabilities-panel">
                        <h4>Class Distribution</h4>
                        <div className="prob-list">
                          {Object.entries(result.lane_1_specialist.probabilities).map(
                            ([label, prob]) => (
                              <div key={label} className="prob-item">
                                <span className="prob-label">{label}</span>
                                <div className="prob-bar-container">
                                  <div
                                    className="prob-bar-fill"
                                    style={{ width: `${prob * 100}%` }}
                                  ></div>
                                </div>
                                <span className="prob-percent">{(prob * 100).toFixed(1)}%</span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    )}

                    {result.lane_1_specialist.recommended_action && (
                      <div className="recommendation-box">
                        <strong>💡 Next Steps:</strong> {result.lane_1_specialist.recommended_action}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Lane 2: Glaucoma & Cataract */}
              {result.lane_2_generalist && (
                <div className="result-card lane-2-card">
                  <div className="card-header">
                    <div className="lane-badge">🚗 Lane 2</div>
                    <h3>Glaucoma & Cataract Detection</h3>
                    <span className="card-icon">🔍</span>
                  </div>

                  <div className="diagnosis-container">
                    <div
                      className="diagnosis-panel"
                      style={{
                        backgroundColor: getSeverityBg(result.lane_2_generalist.severity),
                        borderLeft: `5px solid ${getSeverityColor(result.lane_2_generalist.severity)}`,
                      }}
                    >
                      <div className="diagnosis-section">
                        <label>Diagnosis</label>
                        <div className="diagnosis-value">
                          {result.lane_2_generalist.diagnosis}
                        </div>
                      </div>

                      <div className="confidence-section">
                        <label>Confidence Score</label>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{
                              width: `${result.lane_2_generalist.confidence * 100}%`,
                              backgroundColor: getSeverityColor(result.lane_2_generalist.severity),
                            }}
                          ></div>
                        </div>
                        <div className="confidence-text">
                          {(result.lane_2_generalist.confidence * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="severity-section">
                        <label>Risk Level</label>
                        <div
                          className="severity-badge"
                          style={{
                            backgroundColor: getSeverityColor(result.lane_2_generalist.severity),
                          }}
                        >
                          {result.lane_2_generalist.severity}
                        </div>
                      </div>
                    </div>

                    {result.lane_2_generalist.probabilities && (
                      <div className="probabilities-panel">
                        <h4>Class Distribution</h4>
                        <div className="prob-list">
                          {Object.entries(result.lane_2_generalist.probabilities).map(
                            ([label, prob]) => (
                              <div key={label} className="prob-item">
                                <span className="prob-label">{label}</span>
                                <div className="prob-bar-container">
                                  <div
                                    className="prob-bar-fill"
                                    style={{ width: `${prob * 100}%` }}
                                  ></div>
                                </div>
                                <span className="prob-percent">{(prob * 100).toFixed(1)}%</span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    )}

                    {result.lane_2_generalist.recommended_action && (
                      <div className="recommendation-box">
                        <strong>💡 Next Steps:</strong> {result.lane_2_generalist.recommended_action}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Combined Risk */}
              {result.combined_risk_level && (
                <div className="combined-card">
                  <h3>🎯 Overall Assessment</h3>
                  <div className="combined-value">{result.combined_risk_level}</div>
                </div>
              )}
            </div>

            {/* New Analysis */}
            <div className="action-buttons" style={{ marginTop: '2rem', marginBottom: '2rem' }}>
              <button className="btn btn-primary" onClick={reset}>
                🔄 Analyze Another Image
              </button>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Multi-Ocular AI | Dual-Lane Detection System | v2.0 | API: localhost:8001</p>
      </footer>
    </div>
  );
}
