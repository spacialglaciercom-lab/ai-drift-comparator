# 📊 AI Drift Comparator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-drift-comparator.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/spacialglaciercom-lab/ai-drift-comparator/actions/workflows/ci.yml/badge.svg)](https://github.com/spacialglaciercom-lab/ai-drift-comparator/actions/workflows/ci.yml)

A comprehensive Streamlit application for comparing data drift, model drift, and feature importance drift between reference (training) and current (production) datasets using Evidently AI and Deepchecks.

## ✨ Features

### 🔍 Multi-Dataset Comparison
- Compare 3+ datasets side-by-side
- Heatmap visualization of drift scores across datasets
- Comprehensive comparison tables

### 📊 Three Analysis Modes
- **Data Drift**: Detect distribution shifts using Evidently AI and Deepchecks
- **Model Drift**: Analyze performance decay across multiple models (RandomForest, XGBoost, LogisticRegression)
- **Feature Importance Drift**: Track feature importance changes over time

### 📈 Advanced Visualizations
- **Heatmap**: Drift scores per feature across datasets
- **Waterfall Chart**: Cumulative drift contribution
- **Time-Series Drift**: Automatic detection for temporal data
- **Distribution Comparisons**: Overlay histograms and bar charts

### 🚀 Batch Mode
- Compare multiple models simultaneously
- Side-by-side performance metrics
- Performance drop visualization

### 📥 Export Capabilities
- **JSON**: Complete drift results
- **CSV**: Drifted features list
- **HTML**: Evidently AI comprehensive reports

### 🔔 Alert System
- Red/Yellow/Green badges based on drift severity
- Configurable thresholds
- Webhook support for Slack/Email notifications

### 🎨 UI/UX Enhancements
- **Dark/Light Theme Toggle**: Switch between themes
- **Progress Bars**: Real-time computation progress
- **Shareable Links**: Generate links to share configurations
- **Edge Case Handling**: Validates empty/mismatched files, handles large files

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/spacialglaciercom-lab/ai-drift-comparator.git
cd ai-drift-comparator

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy Commands

#### Streamlit Cloud (Recommended)

```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main

# 2. Go to https://share.streamlit.io
# 3. Click "New app"
# 4. Select repository: spacialglaciercom-lab/ai-drift-comparator
# 5. Set main file: app.py
# 6. Click "Deploy"
```

#### Heroku

```bash
# Install Heroku CLI first: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# Open app
heroku open
```

#### Docker

```bash
# Build image
docker build -t ai-drift-comparator .

# Run container
docker run -p 8501:8501 ai-drift-comparator

# Or with docker-compose
docker-compose up
```

#### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8502

# Run tests
pytest tests/ -v
```

## 📖 Usage

### 1. Data Loading

**Option A: Upload CSV Files**
- Upload reference (training) and current (production) datasets
- Supports multi-dataset comparison mode (3+ datasets)
- Automatic handling of large files (>10MB)

**Option B: Synthetic Data**
- Generate synthetic classification datasets
- Generate adult income-like datasets
- **Demo Data (Normal + Drift)**: Reference with normal distribution, current with 30% features drifted (mean + std*0.5)

### 2. Data Drift Analysis

1. Navigate to the **Data Drift** tab
2. Select detection method (Evidently AI, Deepchecks, or Both)
3. Configure drift threshold
4. View results:
   - Dataset-level drift metrics
   - Feature-level drift scores
   - Top drifting features visualization
   - Distribution comparisons
   - Waterfall charts

### 3. Model Drift Analysis

1. Navigate to the **Model Drift** tab
2. Enable batch mode to compare multiple models
3. Select models (RandomForest, XGBoost, LogisticRegression)
4. View performance metrics:
   - Accuracy/Precision/Recall/F1 for classification
   - MSE/R² for regression
   - Performance drop visualization

### 4. Feature Importance Drift

1. Navigate to the **Feature Importance Drift** tab
2. Select models for analysis
3. Compare feature importance between reference and current datasets
4. View importance drift scores

### 5. Export Results

- Click export buttons to download:
  - JSON metrics
  - CSV of drifted features
  - HTML report (Evidently)

## 🏗️ Project Structure

```
ai-drift-comparator/
├── app.py                      # Main Streamlit application
├── streamlit_app.py            # Entrypoint for Streamlit Cloud
├── requirements.txt            # Full dependencies
├── requirements-deploy.txt     # Streamlit Cloud optimized
├── requirements-dev.txt      # Development dependencies
├── Procfile                   # Heroku/Streamlit Cloud deployment
├── runtime.txt                # Python version specification
├── Dockerfile                 # Docker container definition
├── .dockerignore              # Docker ignore patterns
├── pyproject.toml             # Python project configuration
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── drift_utils.py        # Drift detection utilities
│   └── ui_utils.py           # UI utility functions
├── data/
│   ├── __init__.py
│   ├── sample_data.csv       # Sample dataset
│   └── generate_sample_data.py # Data generation utilities
├── tests/
│   ├── __init__.py
│   └── test_drift_detection.py # Comprehensive drift tests
├── .github/
│   └── workflows/
│       ├── ci.yml            # CI pipeline
│       ├── cd.yml            # CD pipeline
│       └── streamlit.yml     # Streamlit Cloud deployment
├── README.md
└── LICENSE
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_drift_detection.py -v

# Run with coverage
pytest tests/ -v --cov=utils --cov-report=html

# Run with detailed output
pytest tests/ -v -s
```

### Test Coverage

The test suite includes:
- ✅ Drift detection with Evidently AI and Deepchecks
- ✅ Model drift calculation
- ✅ Multi-dataset comparison
- ✅ Edge cases (empty files, mismatched columns, large files)
- ✅ Feature utility functions
- ✅ Demo data generation (normal + 30% drift)

## 🚢 Deployment

### Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository
5. Set main file path to `app.py`
6. Deploy!

### Heroku

```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

### Docker

```bash
docker build -t ai-drift-comparator .
docker run -p 8501:8501 ai-drift-comparator
```

## 📊 API Usage

### Programmatic Drift Detection

```python
from utils.drift_utils import compare_drift, generate_report
import pandas as pd

# Load your datasets
reference_data = pd.read_csv('reference.csv')
current_data = pd.read_csv('current.csv')

# Detect drift
drift_result = compare_drift(
    reference_data,
    current_data,
    method='evidently',
    threshold=0.05
)

# Generate comprehensive report
report = generate_report(
    reference_data,
    current_data,
    threshold=0.05
)

print(f"Dataset drifted: {drift_result['dataset_drifted']}")
print(f"Drifted features: {drift_result['number_of_drifted_features']}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/spacialglaciercom-lab/ai-drift-comparator.git
cd ai-drift-comparator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Evidently AI](https://www.evidentlyai.com/) for drift detection capabilities
- [Deepchecks](https://www.deepchecks.com/) for validation tools
- [Streamlit](https://streamlit.io/) for the amazing framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the ML community**
