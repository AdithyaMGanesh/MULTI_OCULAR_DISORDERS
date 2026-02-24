import React, { useState } from 'react';
import axios from 'axios';

// API endpoint (Dual-Model Backend running on port 8001)
const API_BASE_URL = 'http://localhost:8001';

export default function AppDualModel() {
  // State management
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
      if (!validTypes.includes(file.type)) {
        setError('❌ Please select a valid image file (PNG, JPG, BMP)');
        setSelectedFile(null);
        setPreview(null);
        return;
      }

      if (file.size > 10 * 1024 * 1024) {
        setError('❌ File size too large. Maximum 10MB allowed.');
        setSelectedFile(null);
        setPreview(null);
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

  // Handle drag and drop
  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect({ target: { files } });
    }
  };

  // Handle prediction
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
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success') {
        setResult(response.data);
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Unknown error occurred';
      setError(`❌ Analysis failed: ${errorMsg}`);
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Get color based on severity (Lane 1: DR)
  const getDRColor = () => {
    if (!result || !result.lane_1_specialist) return '#3b82f6';
    const severity = result.lane_1_specialist.severity;
    if (severity === 'Low Risk') return '#10b981';    // Green
    if (severity === 'Medium Risk') return '#f59e0b';  // Amber
    if (severity === 'High Risk') return '#dc2626';    // Red
    return '#3b82f6';
  };

  // Get color based on severity (Lane 2: Glaucoma/Cataract)
  const getGCColor = () => {
    if (!result || !result.lane_2_generalist) return '#3b82f6';
    const severity = result.lane_2_generalist.severity;
    if (severity === 'Low Risk') return '#10b981';    // Green
    if (severity === 'Medium Risk') return '#f59e0b';  // Amber
    if (severity === 'High Risk') return '#dc2626';    // Red
    return '#3b82f6';
  };

  // Determine primary diagnosis - ONLY show diseases that are actually detected
  const getPrimaryDiagnosis = () => {
    if (!result) return null;
    
    const lane1 = result.lane_1_specialist;
    const lane2 = result.lane_2_generalist;
    
    // Priority 1: Check for DR (if it's not "No DR")
    if (lane1 && lane1.diagnosis && lane1.diagnosis !== 'No DR') {
      return { type: 'dr', data: lane1, color: getDRColor() };
    }
    
    // Priority 2: Check for Glaucoma/Cataract (if it's not "Normal")
    if (lane2 && lane2.diagnosis && lane2.diagnosis !== 'Normal') {
      return { type: 'gc', data: lane2, color: getGCColor() };
    }
    
    // Priority 3: If everything is normal, show the normal result
    if (lane1 && lane1.diagnosis === 'No DR') {
      return { type: 'normal', data: lane1, color: '#10b981' };
    }
    
    return null;
  };

  // Get color for combined risk
  const getCombinedRiskColor = () => {
    if (!result) return '#3b82f6';
    if (result.combined_risk_level.includes('HIGH')) return '#dc2626';  // Red
    if (result.combined_risk_level.includes('MEDIUM')) return '#f59e0b'; // Amber
    if (result.combined_risk_level.includes('LOW')) return '#10b981';    // Green
    return '#3b82f6';
  };

  // Styles
  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
    header: {
      background: 'rgba(255, 255, 255, 0.95)',
      padding: '30px 20px',
      borderRadius: '12px',
      marginBottom: '30px',
      boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)',
      textAlign: 'center',
    },
    headerTitle: {
      fontSize: '32px',
      fontWeight: 'bold',
      color: '#1f2937',
      margin: '0 0 10px 0',
    },
    headerSubtitle: {
      fontSize: '14px',
      color: '#6b7280',
      margin: '0',
    },
    mainContent: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '30px',
      maxWidth: '1400px',
      margin: '0 auto',
    },
    column: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
    },
    card: {
      background: 'white',
      borderRadius: '12px',
      padding: '30px',
      boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)',
    },
    uploadArea: {
      border: '2px dashed #cbd5e1',
      borderRadius: '12px',
      padding: '40px 20px',
      textAlign: 'center',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      backgroundColor: '#f8fafc',
    },
    uploadAreaHover: {
      borderColor: '#667eea',
      backgroundColor: '#f0f4ff',
    },
    uploadText: {
      fontSize: '16px',
      color: '#6b7280',
      margin: '0 0 10px 0',
    },
    uploadIcon: {
      fontSize: '40px',
      marginBottom: '10px',
    },
    fileInput: {
      display: 'none',
    },
    previewImage: {
      width: '100%',
      maxHeight: '300px',
      objectFit: 'contain',
      borderRadius: '8px',
      marginBottom: '20px',
      border: '1px solid #e5e7eb',
    },
    fileName: {
      fontSize: '14px',
      color: '#6b7280',
      marginBottom: '15px',
      fontWeight: '500',
    },
    button: {
      padding: '12px 24px',
      fontSize: '16px',
      fontWeight: '600',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      width: '100%',
    },
    analyzeButton: {
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      fontSize: '16px',
      fontWeight: '600',
      padding: '14px 28px',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      width: '100%',
      marginTop: '10px',
    },
    analyzeButtonDisabled: {
      opacity: '0.6',
      cursor: 'not-allowed',
    },
    laneCard: (laneColor) => ({
      border: `3px solid ${laneColor}`,
      background: `rgba(${parseInt(laneColor.slice(1, 3), 16)}, ${parseInt(laneColor.slice(3, 5), 16)}, ${parseInt(laneColor.slice(5, 7), 16)}, 0.05)`,
      marginBottom: '20px',
    }),
    laneTitle: (laneColor) => ({
      fontSize: '20px',
      fontWeight: 'bold',
      color: laneColor,
      margin: '0 0 15px 0',
    }),
    laneDiagnosis: (laneColor) => ({
      fontSize: '28px',
      fontWeight: 'bold',
      color: laneColor,
      marginBottom: '15px',
    }),
    laneLabel: {
      fontSize: '13px',
      color: '#9ca3af',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '10px',
    },
    resultSeverity: {
      fontSize: '14px',
      color: '#6b7280',
      marginBottom: '10px',
      fontWeight: '500',
    },
    confidenceValue: {
      fontSize: '24px',
      fontWeight: 'bold',
      color: '#667eea',
      marginBottom: '15px',
    },
    confidenceLabel: {
      fontSize: '14px',
      color: '#6b7280',
      marginBottom: '20px',
    },
    probabilityItem: {
      marginBottom: '15px',
    },
    probabilityLabel: {
      display: 'flex',
      justifyContent: 'space-between',
      fontSize: '13px',
      fontWeight: '500',
      color: '#374151',
      marginBottom: '5px',
    },
    probabilityBar: {
      height: '8px',
      backgroundColor: '#e5e7eb',
      borderRadius: '4px',
      overflow: 'hidden',
    },
    probabilityFill: (percentage) => ({
      height: '100%',
      background: `linear-gradient(90deg, #667eea 0%, #764ba2 100%)`,
      width: `${percentage}%`,
      transition: 'width 0.5s ease',
    }),
    recommendationBox: {
      background: '#f0f9ff',
      border: '1px solid #7dd3fc',
      borderRadius: '8px',
      padding: '12px',
      marginTop: '15px',
      fontSize: '13px',
      color: '#0369a1',
      lineHeight: '1.5',
    },
    errorBox: {
      background: '#fee2e2',
      border: '1px solid #fca5a5',
      borderRadius: '8px',
      padding: '12px',
      color: '#991b1b',
      fontSize: '14px',
    },
    loadingSpinner: {
      display: 'inline-block',
      width: '20px',
      height: '20px',
      border: '3px solid #f3f4f6',
      borderTop: '3px solid #667eea',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite',
    },
    noResultsText: {
      color: '#9ca3af',
      fontSize: '14px',
      textAlign: 'center',
      padding: '40px 20px',
    },
    combinedRiskBox: (color) => ({
      background: 'rgba(255, 255, 255, 0.95)',
      border: `2px solid ${color}`,
      borderRadius: '12px',
      padding: '20px',
      marginBottom: '20px',
      textAlign: 'center',
    }),
    combinedRiskTitle: (color) => ({
      fontSize: '18px',
      fontWeight: 'bold',
      color: color,
      margin: '0 0 10px 0',
    }),
    combinedRiskText: (color) => ({
      fontSize: '14px',
      color: color,
      fontWeight: '600',
    }),
    gradcamContainer: {
      marginTop: '20px',
      marginBottom: '20px',
      borderRadius: '8px',
      overflow: 'hidden',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    },
    gradcamTitle: {
      fontSize: '13px',
      fontWeight: '600',
      color: '#374151',
      marginBottom: '10px',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    },
    gradcamImage: {
      width: '100%',
      maxWidth: '400px',
      height: 'auto',
      borderRadius: '8px',
      border: '2px solid #e5e7eb',
    },
    gradcamCaption: {
      fontSize: '11px',
      color: '#9ca3af',
      marginTop: '8px',
      fontStyle: 'italic',
    },
  };

  return (
    <div style={styles.container}>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @media (max-width: 768px) {
          body {
            margin: 0;
            padding: 0;
          }
        }
      `}</style>

      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.headerTitle}>🚗🚗 Multi-Ocular Detection Dashboard</h1>
        <p style={styles.headerSubtitle}>Two-Lane Highway: Dual-Model AI System for Comprehensive Eye Disease Detection</p>
      </div>

      {/* Main Content */}
      <div style={styles.mainContent}>
        {/* Left Column: Upload & Control */}
        <div style={styles.column}>
          {/* Upload Area Card */}
          <div style={styles.card}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', color: '#1f2937', margin: '0 0 15px 0' }}>
              📁 Upload Retinal Image
            </h2>
            <div
              style={styles.uploadArea}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => document.getElementById('fileInput').click()}
            >
              <div style={styles.uploadIcon}>📷</div>
              <p style={styles.uploadText}>
                <strong>Click to upload</strong> or drag and drop
              </p>
              <p style={{ fontSize: '12px', color: '#9ca3af', margin: '0' }}>
                PNG, JPG, BMP • Max 10MB
              </p>
            </div>
            <input
              id="fileInput"
              type="file"
              style={styles.fileInput}
              onChange={handleFileSelect}
              accept="image/*"
            />

            {/* Preview */}
            {preview && (
              <>
                <img src={preview} alt="Preview" style={styles.previewImage} />
                <div style={styles.fileName}>
                  ✅ {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                </div>
              </>
            )}

            {/* Error Message */}
            {error && <div style={styles.errorBox}>{error}</div>}

            {/* Analyze Button */}
            <button
              style={{
                ...styles.analyzeButton,
                ...(loading || !selectedFile ? styles.analyzeButtonDisabled : {}),
              }}
              onClick={handleAnalyze}
              disabled={loading || !selectedFile}
            >
              {loading ? (
                <>
                  <span style={styles.loadingSpinner}></span> Analyzing (Both Lanes)...
                </>
              ) : (
                '⚡ Analyze Image (Dual Model)'
              )}
            </button>
          </div>

          {/* Info Card */}
          {!result && !loading && (
            <div style={styles.card}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#1f2937', margin: '0 0 12px 0' }}>
                ℹ️ System Overview
              </h3>
              <div style={{ fontSize: '13px', color: '#6b7280', lineHeight: '1.8', margin: '0' }}>
                <p><strong>🚗 Lane 1 (Specialist):</strong> Diabetic Retinopathy Detection (224×224)</p>
                <p><strong>🚗 Lane 2 (Generalist):</strong> Glaucoma/Cataract Detection (128×128)</p>
                <p><strong>⚙️ Processing:</strong> Both models run in parallel for comprehensive diagnosis</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Results */}
        <div style={styles.column}>
          {result ? (
            <>
              {(() => {
                const primary = getPrimaryDiagnosis();
                
                if (!primary) {
                  return (
                    <div style={styles.card}>
                      <p style={styles.noResultsText}>Unable to process results</p>
                    </div>
                  );
                }

                const { type, data, color } = primary;
                const diseaseTitle = type === 'dr' ? 'Diabetic Retinopathy' : type === 'gc' ? data.disease : 'Eye Health Status';
                const isAbnormal = type === 'dr' ? data.diagnosis !== 'No DR' : type === 'gc' ? data.diagnosis !== 'Normal' : false;

                return (
                  <>
                    {/* Primary Diagnosis Card */}
                    <div style={{ ...styles.card, borderLeft: `5px solid ${color}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                        <h2 style={{ ...styles.laneTitle(color), margin: 0 }}>
                          {isAbnormal ? '⚠️' : '✅'} {diseaseTitle}
                        </h2>
                      </div>

                      {/* Classification */}
                      <div style={{
                        ...styles.laneDiagnosis(color),
                        fontSize: '28px',
                        marginBottom: '20px',
                        textAlign: 'center',
                        padding: '15px',
                        background: `${color}15`,
                        borderRadius: '8px',
                      }}>
                        {data.diagnosis}
                      </div>

                      {/* Severity & Confidence in Row */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '20px' }}>
                        <div style={{ padding: '10px', background: '#f3f4f6', borderRadius: '6px' }}>
                          <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>SEVERITY</div>
                          <div style={{ fontSize: '16px', color: color, fontWeight: 'bold', marginTop: '5px' }}>
                            {data.severity}
                          </div>
                        </div>
                        <div style={{ padding: '10px', background: '#f3f4f6', borderRadius: '6px' }}>
                          <div style={{ fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>CONFIDENCE</div>
                          <div style={{ fontSize: '20px', color: color, fontWeight: 'bold', marginTop: '5px' }}>
                            {(data.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {/* GradCAM Visualization */}
                      {data.gradcam && (
                        <div style={styles.gradcamContainer}>
                          <div style={styles.gradcamTitle}>🔥 Attention Heatmap (Grad-CAM)</div>
                          <img
                            src={`data:image/png;base64,${data.gradcam}`}
                            alt="Grad-CAM Heatmap"
                            style={styles.gradcamImage}
                          />
                          <div style={styles.gradcamCaption}>
                            Red areas indicate regions contributing most to this diagnosis
                          </div>
                        </div>
                      )}

                      {/* Probability Distribution */}
                      {data.probabilities && Object.keys(data.probabilities).length > 0 && (
                        <div style={{ marginTop: '20px', marginBottom: '20px' }}>
                          <h4 style={{ fontSize: '13px', fontWeight: '600', color: '#1f2937', margin: '0 0 15px 0' }}>
                            📊 Classification Probabilities
                          </h4>
                          {Object.entries(data.probabilities).map(([className, probability]) => (
                            <div key={className} style={styles.probabilityItem}>
                              <div style={styles.probabilityLabel}>
                                <span>{className}</span>
                                <span>{(probability * 100).toFixed(1)}%</span>
                              </div>
                              <div style={styles.probabilityBar}>
                                <div style={styles.probabilityFill(probability * 100)} />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Recommendation */}
                      {data.recommended_action && (
                        <div style={styles.recommendationBox}>
                          <strong>📋 Recommended Action:</strong>
                          <div style={{ marginTop: '8px' }}>{data.recommended_action}</div>
                        </div>
                      )}
                    </div>

                    {/* Secondary Info Button */}
                    <details style={{ marginTop: '15px', marginBottom: '15px' }}>
                      <summary style={{
                        cursor: 'pointer',
                        padding: '10px',
                        background: '#f3f4f6',
                        borderRadius: '6px',
                        fontWeight: '600',
                        color: '#374151',
                      }}>
                        📋 View Full Analysis (Both Lanes)
                      </summary>
                      <div style={{ marginTop: '15px' }}>
                        {/* Lane 1 Details */}
                        {result.lane_1_specialist && (
                          <div style={{ ...styles.card, marginBottom: '15px', opacity: 0.8 }}>
                            <div style={styles.laneLabel}>🚗 Lane 1 - Specialist (DR Detection)</div>
                            <div style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
                              <div>Disease: <strong>{result.lane_1_specialist.disease}</strong></div>
                              <div>Diagnosis: <strong>{result.lane_1_specialist.diagnosis}</strong></div>
                              <div>Confidence: <strong>{(result.lane_1_specialist.confidence * 100).toFixed(1)}%</strong></div>
                            </div>
                          </div>
                        )}

                        {/* Lane 2 Details */}
                        {result.lane_2_generalist && (
                          <div style={{ ...styles.card, opacity: 0.8 }}>
                            <div style={styles.laneLabel}>🚗 Lane 2 - Generalist (Glaucoma/Cataract)</div>
                            <div style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
                              <div>Disease: <strong>{result.lane_2_generalist.disease}</strong></div>
                              <div>Diagnosis: <strong>{result.lane_2_generalist.diagnosis}</strong></div>
                              <div>Confidence: <strong>{(result.lane_2_generalist.confidence * 100).toFixed(1)}%</strong></div>
                            </div>
                          </div>
                        )}

                        {/* Combined Risk */}
                        {result.combined_risk_level && (
                          <div style={{
                            ...styles.card,
                            background: '#f0f9ff',
                            border: '1px solid #7dd3fc',
                            marginTop: '15px'
                          }}>
                            <div style={{ fontSize: '13px', color: '#0369a1', fontWeight: '600', marginBottom: '8px' }}>
                              COMBINED RISK LEVEL
                            </div>
                            <div style={{ fontSize: '14px', color: '#0369a1', fontWeight: 'bold' }}>
                              {result.combined_risk_level}
                            </div>
                          </div>
                        )}
                      </div>
                    </details>
                  </>
                );
              })()}

              {/* Reset Button */}
              <button
                style={{
                  ...styles.button,
                  background: '#f3f4f6',
                  color: '#374151',
                  marginTop: '20px',
                }}
                onClick={() => {
                  setResult(null);
                  setSelectedFile(null);
                  setPreview(null);
                  setError(null);
                }}
              >
                🔄 Analyze Another Image
              </button>
            </>
          ) : (
            <div style={styles.card}>
              <div style={styles.noResultsText}>
                {loading ? (
                  <>
                    <div style={{ ...styles.loadingSpinner, margin: '20px auto' }}></div>
                    <p style={{ marginTop: '20px' }}>Running dual-model analysis...</p>
                    <p style={{ fontSize: '12px' }}>Both lanes processing your image</p>
                  </>
                ) : (
                  <>
                    <p style={{ fontSize: '16px', marginBottom: '10px' }}>👈 Upload an image to get started</p>
                    <p style={{ fontSize: '12px' }}>Results from both lanes will appear here</p>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div
        style={{
          textAlign: 'center',
          marginTop: '40px',
          color: 'rgba(255, 255, 255, 0.7)',
          fontSize: '12px',
        }}
      >
        <p>
          🩺 Multi-Ocular Two-Lane Highway Detection System v2.0 | Dual-Model AI-Powered Diagnosis
        </p>
      </div>
    </div>
  );
}
