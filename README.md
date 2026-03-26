# AeroNet RUL: Hybrid CAE-BiLSTM for Turbofan Prognostics

🚀 **Live Deployment:** [aeronetrul-raquel.streamlit.app](https://aeronetrul-raquel.streamlit.app)

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Project Architecture](#-project-architecture)
3. [Technical Specifications](#-technical-specifications)
4. [Methodology & Data Strategy](#-methodology--data-strategy)
5. [Key Performance Indicators](#-key-performance-indicators)
6. [Deployment & MLOps](#-deployment--mlops)
7. [Contact & Verification](#-contact--verification)

---

## ✈️ Executive Summary
**AeroNet RUL** is a high-fidelity deep learning framework developed for **Prognostics and Health Management (PHM)**. By leveraging the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset, this project demonstrates a scalable approach to forecasting the **Remaining Useful Life (RUL)** of aerospace assets to ensure fleet readiness and mission-critical reliability [1].

## 🧠 Project Architecture
The proposed system utilizes a proprietary hybrid architecture designed for high-dimensional sensor telemetry:

*   **Feature Extraction:** A Convolutional-inspired (CAE) backbone utilizing **Conv1D layers** with custom padding to maintain temporal resolution without dimensionality loss.
*   **Sequential Intelligence:** A stacked **Bidirectional LSTM (Bi-LSTM)** framework to capture long-range temporal dependencies across engine degradation cycles.
*   **Dynamic Weighting:** An integrated **Temporal Attention Mechanism** (Softmax-driven) that dynamically prioritizes critical sensor states during high-stress flight regimes.
*   **Regularization:** High-capacity Dense layers with **Dropout regularization** and GlobalAveragePooling1D to ensure robust feature extraction and prevent over-fitting.

## ⚙️ Technical Specifications
*   **Core Framework:** Keras 3 (Backend-agnostic: TensorFlow/PyTorch/JAX)
*   **Data Processing:** Pandas, NumPy (Optimized for time-series sliding windows)
*   **Visualization:** Plotly (Interactive Aerospace Telemetry Dashboard)
*   **Cloud Infrastructure:** Streamlit Cloud (Python 3.11 Environment)

## 📊 Methodology & Data Strategy
*   **Data Source:** NASA C-MAPSS Turbofan Degradation Dataset.
*   **Normalization:** Min-Max scaling fit to baseline nominal engine states.
*   **Windowing:** Sliding-window temporal segmentation to preserve historical state during real-time inference.
*   **Compatibility:** Optimized for **Keras 3** to ensure stable performance across diverse hardware profiles.

## 📈 Key Performance Indicators
- **Operational Efficiency:** Automated maintenance scheduling via high-confidence RUL forecasting.
- **Explainability:** Visualized attention weights showing which sensors (e.g., Pressure, Temperature, Fan Speed) most influence the degradation curve.
- **Resource Optimization:** Optimized for **Edge Inference** on resource-constrained hardware (e.g., flight simulator modules).

## 🚀 Deployment & MLOps
- **CI/CD:** Automated deployment via GitHub synchronization.
- **Version Control:** Managed via Git (Commit history reflects iterative R&D and conflict resolution).
- **Environment Management:** Pinning of exact manylinux wheels (NumPy 1.26) for stable cloud-native execution.

## 📝 Author
**Raquel Adams, Ph.D.**  



