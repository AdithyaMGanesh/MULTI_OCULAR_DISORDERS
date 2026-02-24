import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="nav-content">
          <div className="nav-logo">
            <span className="logo-icon">👁️</span>
            <span className="logo-text">Multi-Ocular</span>
          </div>
          <div className="nav-links">
            <a href="#features" className="nav-link">Features</a>
            <a href="#about" className="nav-link">About</a>
            <a href="#contact" className="nav-link">Contact</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">AI-Powered Eye Disease Detection</h1>
            <p className="hero-subtitle">
              Advanced dual-model system for detecting Diabetic Retinopathy, Glaucoma, and Cataract
            </p>
            <p className="hero-description">
              Using cutting-edge deep learning with VGG16, ResNet50, and DenseNet121 fusion architecture
            </p>
            <button
              className="cta-button"
              onClick={() => navigate('/predict')}
            >
              🚀 Start Analysis
            </button>
          </div>
          <div className="hero-image">
            <div className="image-placeholder">
              <span className="large-icon">👁️</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features" id="features">
        <div className="features-header">
          <h2>Why Choose Multi-Ocular?</h2>
          <p>Advanced AI technology for accurate eye disease detection</p>
        </div>

        <div className="features-grid">
          {/* Feature 1: Dual Models */}
          <div className="feature-card">
            <div className="feature-icon">🚗</div>
            <h3>Dual-Lane Detection</h3>
            <p>
              Two specialized models running in parallel - Lane 1 for Diabetic Retinopathy and Lane 2 for Glaucoma/Cataract detection
            </p>
          </div>

          {/* Feature 2: High Accuracy */}
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3>High Accuracy</h3>
            <p>
              State-of-the-art deep learning models trained on large fundus image datasets for reliable predictions
            </p>
          </div>

          {/* Feature 3: Fast Processing */}
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Fast Processing</h3>
            <p>
              GPU-optimized inference delivering results in seconds for rapid clinical decision support
            </p>
          </div>

          {/* Feature 4: Professional UI */}
          <div className="feature-card">
            <div className="feature-icon">✨</div>
            <h3>Professional Interface</h3>
            <p>
              Intuitive drag-and-drop upload with detailed analytics and severity risk assessment
            </p>
          </div>

          {/* Feature 5: Detailed Reports */}
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Detailed Reports</h3>
            <p>
              Comprehensive results with confidence scores, probability distributions, and clinical recommendations
            </p>
          </div>
     

        </div>
      </section>

      {/* Models Section */}
      <section className="models">
        <div className="models-header">
          <h2>Advanced AI Models</h2>
          <p>Powered by state-of-the-art deep learning architectures</p>
        </div>

        <div className="models-grid">
          {/* Lane 1 */}
          <div className="model-card lane-1-card">
            <div className="model-badge">🚗 Lane 1</div>
            <h3>Diabetic Retinopathy Specialist</h3>
            <div className="model-details">
              <div className="detail-item">
                <span className="detail-label">Model:</span>
                <span className="detail-value">Fusion Architecture</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Base Models:</span>
                <span className="detail-value">VGG16 + ResNet50 + DenseNet121</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Input Size:</span>
                <span className="detail-value">224×224 pixels</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Classes:</span>
                <span className="detail-value">5 (No DR, Mild, Moderate, Severe, Proliferative)</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Parameters:</span>
                <span className="detail-value">54.3M</span>
              </div>
            </div>
          </div>

          {/* Lane 2 */}
          <div className="model-card lane-2-card">
            <div className="model-badge">🚗 Lane 2</div>
            <h3>Glaucoma & Cataract Generalist</h3>
            <div className="model-details">
              <div className="detail-item">
                <span className="detail-label">Model:</span>
                <span className="detail-value">Lightweight CNN</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Architecture:</span>
                <span className="detail-value">Custom Convolutional Network</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Input Size:</span>
                <span className="detail-value">128×128 pixels</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Classes:</span>
                <span className="detail-value">3 (Normal, Glaucoma, Cataract)</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Parameters:</span>
                <span className="detail-value">94.5K</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Ready to Analyze Eye Images?</h2>
          <p>Start using our AI system for accurate eye disease detection</p>
          <button
            className="cta-button large"
            onClick={() => navigate('/predict')}
          >
            🔬 Go to Prediction
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Multi-Ocular</h4>
            <p>AI-Powered Eye Disease Detection System</p>
          </div>
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="#features">Features</a></li>
              <li><a href="#about">About</a></li>
              <li><a href="/predict">Analyzer</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Technology</h4>
            <ul>
              <li>Deep Learning</li>
              <li>Computer Vision</li>
              <li>Medical AI</li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2024 Multi-Ocular AI System. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
