# Student Service Request Monitoring (Streamlit)

Deploys the midterm Decision Tree model that predicts **SLA breaches** for student
service requests, with a data/EDA module, prediction form, decision-support dashboard
and SAD documentation.

> Predictions are decision support only. Staff review every flagged request and make
> the final decision. Use synthetic or properly authorized data. Never commit real
> student records.

## Pages
Home | Dataset | Data Analysis | Prediction | Dashboard | System Design

## Project structure
```
app.py                          Streamlit entry point (navigation, Home, System Design)
train_model.py                  Trains and saves the pipeline (run once, not on refresh)
requirements.txt
modules/data_processing.py      CSV loading, schema checks, cleaning, features
modules/exploratory_analysis.py Interactive Plotly EDA
modules/prediction.py           Loads the saved model, inference, metrics, consistency check
modules/dashboard.py            KPIs, risk/review queue, recorded human decisions
modules/model_training.py       Training code used by train_model.py
models/trained_model.joblib     Saved preprocessing + Decision Tree pipeline
models/model_info.json          Version, timestamp, metrics, dropdown options
models/test_cases.csv           Held-out cases used by the consistency check
assets/use_case_diagram.png     Updated use case diagram
assets/decision_workflow.png    Analytical process flow / decision workflow
data/sample_requests.csv        SYNTHETIC sample data (same schema as the midterm)
tools/                          Scripts that made the sample data and diagrams
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## IMPORTANT: use your own midterm CSV
The bundled `models/trained_model.joblib` was trained on **synthetic** data so the app
works out of the box. To use your midterm results:
```
python train_model.py path/to/your_midterm_dataset.csv
```
This uses the same settings as the midterm notebook (Decision Tree, max_depth=5,
min_samples_leaf=5, random_state=42, 80/20 stratified split), so it should reproduce
your midterm metrics. Commit the new files in `models/`. Do not commit the CSV itself
if it has real student data.

If the script prints a different scikit-learn version, update that line in
`requirements.txt` to match.

## Deploy on Streamlit Community Cloud
1. Push everything to a GitHub repository.
2. Go to https://share.streamlit.io and sign in.
3. Create app -> choose the repository and branch -> main file path `app.py` -> Deploy.
4. If the build fails, open the logs (usually a version or missing-file problem).

## Test checklist
- [ ] Deployed URL opens without an error
- [ ] Dataset: valid CSV, missing columns, empty file, non-CSV file
- [ ] Data Analysis: filters and all six charts
- [ ] Prediction: valid inputs, five sample predictions, metrics, consistency check
- [ ] Dashboard: KPIs, review queue, record a decision (reviewer name required)
- [ ] System Design: requirements, use case diagram, workflow image
