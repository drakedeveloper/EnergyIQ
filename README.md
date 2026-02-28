# EnergyIQ: Smart City Energy Management Platform

EnergyIQ is an AI-powered platform for real-time monitoring, forecasting, anomaly detection, and actionable recommendations for city energy consumption. This project is a full-stack application with a Python FastAPI backend (using a Gated Graph Convolutional Network, PyTorch) and a modern React frontend dashboard.

## Features
- Real-time energy consumption monitoring for city zones
- AI-based forecasting (GGCN, PyTorch)
- Anomaly detection (error-based, GGCN)
- Actionable energy-saving recommendations (data-driven)
- Interactive dashboards and visualizations (React)
- Feature importance and model explainability (MIAF)

---


## Included in this Repository

- 📓 **Jupyter Notebook**: You will find a notebook in the repo that demonstrates data exploration, model training, and evaluation steps. This is useful for understanding the workflow and for reproducibility.
- 🖼️ **Visualizations**: The following images are included:
   - `1.png` and `2.png`: Show result visualizations (e.g., predictions, error analysis, or dashboard screenshots).
   - `3.png`: Shows the MIAF + GGCN architecture for RUL (Remaining Useful Life) prediction.

---

## Project Structure

```
backend/
   data.py
   main.py
   model.py
   recommendations.py
   requirements.txt
   data/
      smart_city_energy_dataset.csv
      ggcn_rul_model.pth
      miaf_ggcn_model_info.json
frontend/
   package.json
   public/
      index.html
   src/
      App.js
      ...
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+


### Backend Setup
1. Navigate to the backend folder:
   ```sh
   cd backend
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```sh
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup
1. Navigate to the frontend folder:
   ```sh
   cd frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the React development server:
   ```sh
   npm start
   ```
   The app will run at http://localhost:3000

#### Environment Variables
- The frontend uses a `.env` file to set the API URL:
  ```env
  REACT_APP_API_URL=http://localhost:8000
  ```
  Change this if your backend runs on a different address.

---

## Usage
- Open http://localhost:3000 in your browser.
- Ensure the backend is running at the address specified in the `.env` file.
- Explore the dashboard tabs: Overview, Forecast, Anomalies, AI Tips, Features.

---

## API Endpoints (Backend)

- `GET /` — API status
- `GET /api/realtime` — Real-time grid data and prediction
- `GET /api/forecast?hours=48` — Forecast for next N hours
- `GET /api/anomalies` — Recent detected anomalies
- `GET /api/evaluation` — Model evaluation metrics
- `GET /api/recommendations` — AI-driven energy-saving tips
- `GET /api/features` — Feature importance and model info

---

## Data
- The backend uses `data/smart_city_energy_dataset.csv` as the main dataset for modeling and analytics.
- Model weights: `data/ggcn_rul_model.pth`
- Feature info: `data/miaf_ggcn_model_info.json`

---

## Technologies
- **Backend:** Python, FastAPI, PyTorch, scikit-learn, pandas
- **Frontend:** React, recharts, axios, leaflet

---

## License
This project is for educational and demonstration purposes only.
