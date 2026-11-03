# ☄️ Transient-Finder: Near-Earth Object (NEO) & Sky Survey AI Validation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An edge-optimized, AI-driven astronomical analysis system engineered to ingest, evaluate, and cross-validate unconfirmed **Near-Earth Object (NEO)** candidates and sky survey transients from NASA JPL, the Minor Planet Center (MPC), and modern time-domain brokers (ZTF / Fink / LSST).

Designed specifically to run locally on high-performance laptops (e.g. MacBook Pro Intel/Apple Silicon, 16GB RAM) without requiring heavy cloud multi-GPU clusters.

---

## 🌟 Key Capabilities

1. **Multi-Feed Astronomical Ingestion:**
   - **NASA JPL CNEOS Scout API:** Live real-time scoring, hazard ratings, and Monte Carlo orbital covariance samples.
   - **Minor Planet Center (MPC) NEOCP:** Live Near-Earth Object Confirmation Page tracklet ingestion.
   - **Fink Broker (ZTF & Rubin LSST):** Ingestion of freshly linked Solar System Object (SSO) candidates from Fink-FAT.
   - **NASA JPL Close Approach Data (CAD) & SBDB:** Historical candidate recovery and longitudinal tracking for candidates submitted ~6 months ago (e.g., May 2026 objects like `2026 HX3`, `2026 HY3`).
   - **Transparent Disk Cache & Offline Failover:** Pre-seeded with realistic ground truth fixtures and synthetic false positives so researchers can work offline anywhere.

2. **Five-Pillar AI Heuristic Scoring Engine:**
   - **Astrometric Consistency (30%):** Evaluates tracklet linearity ($R^2$), residual RMS ($\le 0.30''$), and topocentric acceleration constraints.
   - **Orbit Feasibility (25%):** Assesses Keplerian motion bounds, Jacobi energy, and the Tisserand parameter with respect to Jupiter ($T_J > 3.0$ for asteroidal orbits).
   - **Observational Reliability (20%):** Arc length coverage, independent observatory verification, and twilight glare suppression.
   - **Satellite & Space Debris Veto (15%):** Rejects artificial satellites and LEO space debris traveling at non-orbital speeds ($>30^\circ/\text{day}$) or in constellation inclination bands ($i \approx 53^\circ$).
   - **Photometric Stability (10%):** Validates apparent magnitude vs. physical absolute magnitude ($H$).

3. **Supervised ML Real-Bogus Classifier & Anomaly Detector:**
   - Gradient Boosting & Random Forest ensemble trained on optical difference-imaging artifacts.
   - Unsupervised `IsolationForest` detector flagging rare interstellar interlopers (e.g., 'Oumuamua, Borisov with $e > 1.2$) or ultra-close Earth grazers.

4. **Root-Cause False-Positive Explainer:**
   - Categorizes bogus alerts into specific astronomical modes:
     - `DIPOLE_SUBTRACTION_ARTIFACT` (PSF misregistration in difference imaging)
     - `ARTIFICIAL_SATELLITE_OR_DEBRIS` (Starlink / LEO / GEO orbital contamination)
     - `BRIGHT_STAR_DIFFRACTION_GHOST` (Diffraction spikes and internal optical ghosts)
     - `SPURIOUS_NOISE_LINKAGE` (Accidental cross-epoch noise alignments)
     - `STATIONARY_VARIABLE_STAR` (Uncatalogued stellar flare / variable star)

5. **6-Month Candidate Re-Validation Lab:**
   - Models ephemeris covariance expansion $\sigma_{\Delta \theta}(t)$ over elapsed months to calculate recovery probability and assess confirmed vs. lost status.

---

## 🚀 Quickstart Guide (Local Laptop Execution)

### 1. Prerequisites & Environment Setup
This repository is configured to run out-of-the-box on macOS / Linux using Python 3.9+ with an isolated virtual environment:

```bash
# Clone the repository and navigate into it
cd Transient-finder

# Activate the virtual environment
source .venv/bin/activate

# Verify dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard (Streamlit)
Run the browser dashboard to visualize live candidates, 3D orbits, heuristic radar charts, and false positive diagnostics:

```bash
export PYTHONPATH=.
./.venv/bin/streamlit run ui/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Command-Line Interface (CLI)
Perform rapid terminal-based audits using the `rich` CLI:

```bash
# Ingest and validate live candidates from NASA JPL Scout & MPC
export PYTHONPATH=.
./.venv/bin/python3 cli/main.py live --limit 10

# Run 6-month historical re-validation audit (e.g. May 2026 close approaches)
./.venv/bin/python3 cli/main.py historical --limit 5

# Run benchmark ground-truth test suite
./.venv/bin/python3 cli/main.py benchmark
```

### 4. Run the REST API Backend (FastAPI)
Launch the microservice API server for automated survey ingestion pipelines:

```bash
export PYTHONPATH=.
./.venv/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at: `http://localhost:8000/docs`.

### 5. Execute Test Suite
Run the full automated test suite (22 unit & integration tests):

```bash
export PYTHONPATH=.
./.venv/bin/pytest tests/
```

---

## 📁 Repository Structure

```
Transient-finder/
├── requirements.txt            # Python 3.9+ pinned dependencies
├── config.py                   # Configuration & physical heuristic thresholds
├── core/
│   ├── models.py               # Pydantic data schemas for candidates & diagnostics
│   ├── astronomy.py            # Coordinate transformations, solar elongation, Tisserand, families
│   └── orbit_evaluator.py      # Polynomial tracklet fitting, residuals, ephemeris drift
├── ingestion/
│   ├── base.py                 # Abstract HTTP client with disk caching & retry
│   ├── jpl_scout.py            # NASA JPL CNEOS Scout API client
│   ├── mpc_neocp.py            # Minor Planet Center NEOCP bulletin feed parser
│   ├── fink_broker.py          # Fink broker client for ZTF/LSST solar system candidates
│   ├── jpl_cad.py              # NASA JPL Close Approach Data & SBDB client
│   └── fixtures.py             # Pre-seeded ground truth NEOs & false-positive cases
├── ai/
│   ├── feature_extractor.py    # Physical & observational feature extraction (20 features)
│   ├── heuristic_engine.py     # 5-Pillar scientific heuristic scoring engine
│   ├── ml_classifier.py        # Gradient Boosting Real-Bogus classifier
│   ├── anomaly_detector.py     # Unsupervised Isolation Forest for interstellar interlopers
│   ├── false_positive_analyzer.py # Root-cause diagnostic classifier
│   ├── revalidation_engine.py  # 6-month longitudinal recovery & drift evaluator
│   └── validator.py            # Master validation orchestrator
├── api/
│   └── server.py               # FastAPI microservice application
├── ui/
│   └── app.py                  # Interactive Streamlit Web UI dashboard
├── cli/
│   └── main.py                 # Rich CLI application
├── tests/                      # 22 automated unit and integration tests
└── docs/                       # Comprehensive scientific and architectural documentation
    ├── RESEARCH_FINDINGS.md    # Detailed research findings document
    ├── SYSTEM_APPROACH.md      # System approach & heuristic methodology document
    └── TECHNICAL_ARCHITECTURE.md # Technical architecture & deployment document
```

---

## 📚 Technical Documentation

- **[Research Findings Document](file:///Users/codenic7/Documents/workspace/Transient-finder/docs/RESEARCH_FINDINGS.md)**: Exhaustive survey of NASA/MPC data feeds, 6-month historical candidates, and classical vs AI algorithms.
- **[System Approach Document](file:///Users/codenic7/Documents/workspace/Transient-finder/docs/SYSTEM_APPROACH.md)**: Deep dive into the 5-pillar heuristic methodology, ML real-bogus integration, and false-positive root causes.
- **[Technical Architecture Document](file:///Users/codenic7/Documents/workspace/Transient-finder/docs/TECHNICAL_ARCHITECTURE.md)**: Software architecture, component contracts, Mermaid diagrams, and edge laptop performance optimizations.
