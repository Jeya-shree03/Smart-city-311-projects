# Smart City 311 Complaint Analytics & Service Prediction

A beginner-friendly Streamlit dashboard using NYC 311 Service Requests from NYC Open Data.

## Features
- Live data fetch from the NYC Open Data API
- Complaint type and borough charts
- Monthly request trend
- Simple machine-learning demo to classify closure time as over 7 days
- Data preview

## Run on Windows

1. Install Python 3.11 or newer from https://www.python.org/downloads/
2. Open this project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Run these commands one at a time:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Your browser should open the dashboard. If it does not, open the local URL shown in the terminal.

## Data source
NYC Open Data: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data

## Important note
This is an educational demonstration. The 7-day threshold is a project-defined label, not an official NYC service standard. The model only trains on records with a recorded closure time.
