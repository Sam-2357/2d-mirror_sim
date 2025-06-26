**Interactive 2‑D visualization and simulation of a single Optotune MR‑E‑2 mirror steering a laser beam**, with standoff path‑length correction.

<img src="https://img.shields.io/badge/plotly-interactive-blue" alt="Plotly interactive"/>

---

## 🚀 Features

- Simulates a **2D optical geometry**: point emitter → reflective mirror → outgoing beam.
- Models **±25° mirror tilt**, **1.3 mm standoff**, and **beam intercept logic**.
- Computes:
  - Mirror normal, strike point, and finite mirror segment.
  - Incoming and reflected unit vectors.
  - Deflection angle α.
  - Standoff path-length correction Δℓ = 2 d sin(α/2).
- Renders **interactive Plotly visualizations** with:
  - Color-coded rays (black, green),
  - Mirror surface (orange),
  - Normal (blue),
  - Standoff vector (red),
  - Hover and annotation display for α and Δℓ.

---

## 📦 Installation

Requires Python 3.8+. Clone and install dependencies:
