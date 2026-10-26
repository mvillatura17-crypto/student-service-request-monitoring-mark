"""Student Service Request Monitoring - SLA breach prediction (Streamlit).

Midterm data-mining outputs deployed as an information-system workflow.
Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.dashboard import render_dashboard_page
from modules.data_processing import get_dataset, render_dataset_page
from modules.exploratory_analysis import render_eda_page
from modules.prediction import (
    load_model_info, model_files_exist, render_prediction_page,
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"

st.set_page_config(page_title="Student Service Request Monitoring", page_icon="📊",
                   layout="wide")

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]

# Results reported in the submitted midterm notebook
MIDTERM = {
    "rows": 400,
    "eligible": 304,
    "train": 243,
    "test": 61,
    "metrics": {"Accuracy": 0.820, "Precision (breach)": 0.833,
                "Recall (breach)": 0.806, "F1-score (breach)": 0.820},
    "confusion_matrix": [[25, 5], [6, 25]],
}


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
def render_home() -> None:
    st.title("📊 Student Service Request Monitoring")
    st.subheader("Predicting SLA breaches to support student services staff")
    st.write(
        "This application deploys the Decision Tree model from the midterm laboratory "
        "examination as a working information system. It lets staff upload request data, "
        "explore delays and workload, predict which requests may breach their "
        "service-level agreement (SLA), and review a risk queue before deciding on "
        "follow-up actions."
    )
    st.info("Predictions are decision-support outputs. A staff member reviews each "
            "flagged request and makes the final decision. Use synthetic or properly "
            "authorized data only.")

    st.markdown("### How to use")
    st.markdown(
        "1. **Dataset** - upload the CSV (or load the sample data) and check validation.\n"
        "2. **Data Analysis** - explore volume, breach rates and class balance.\n"
        "3. **Prediction** - enter a request's details, get a prediction, and view metrics.\n"
        "4. **Dashboard** - see KPIs and the review queue, then record a human decision.\n"
        "5. **System Design** - requirements, actors, use case diagram and workflow."
    )

    st.markdown("### Continuity with the midterm")
    st.table(pd.DataFrame({
        "Item": ["Problem", "Target", "Target rule", "Predictors", "Model", "Prediction point"],
        "Midterm and this app": [
            "Student Service Request Monitoring (Section B)",
            "SLA_Breached (1 = breach, 0 = no breach)",
            "Processing_Days > SLA_Days for completed requests",
            "Service type, requester type, channel, priority, office, documents "
            "complete, submission weekday, queue length, SLA days",
            "Decision Tree (max_depth=5, min_samples_leaf=5, random_state=42)",
            "At submission time (no post-processing fields)",
        ],
    }))
    st.caption("No difference from the midterm problem scope. New in this app: a saved "
               "joblib pipeline, an input form for new predictions, a risk/review "
               "dashboard, and the SAD documentation.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Midterm notebook results**")
        st.write(f"{MIDTERM['rows']} rows, {MIDTERM['eligible']} eligible completed requests "
                 f"({MIDTERM['train']} train / {MIDTERM['test']} test)")
        st.dataframe(pd.Series(MIDTERM["metrics"], name="Midterm").to_frame())
    with c2:
        st.markdown("**Currently deployed model**")
        if model_files_exist():
            info = load_model_info()
            st.write(f"Version {info['model_version']} | trained {info['trained_at']}")
            st.write(f"Data source: {info['data_source']}")
            st.dataframe(pd.Series(info["metrics"], name="Deployed").round(3).to_frame())
            if info["data_source"].lower().startswith("synthetic"):
                st.caption("These numbers differ from the midterm because the bundled "
                           "model was trained on synthetic data. Retrain with the midterm "
                           "CSV using train_model.py to reproduce the midterm metrics.")
        else:
            st.error("Saved model not found in models/.")


# ---------------------------------------------------------------------------
# System Design
# ---------------------------------------------------------------------------
def _show_image(name: str, caption: str) -> None:
    path = ASSETS / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.error(f"Missing image: assets/{name}")


def render_system_design() -> None:
    st.header("SAD Integration")

    st.subheader("Actors and permissions")
    st.table(pd.DataFrame({
        "Actor": ["Administrator", "Service Staff (student services staff)"],
        "Can view results": ["Yes - data analysis, predictions, dashboard",
                             "Yes - data analysis, predictions, dashboard"],
        "Can upload data": ["Yes", "No"],
        "Records final decisions": ["No (oversight only)", "Yes - after reviewing each flagged request"],
    }))
    st.caption("This prototype has no login. The roles above are documented for a future "
               "authenticated version; the reviewer name is typed in when a decision is saved.")

    st.subheader("Functional requirements (evidence from the midterm)")
    st.table(pd.DataFrame({
        "ID": ["FR1", "FR2", "FR3", "FR4", "FR5"],
        "Requirement": [
            "The system shall let users upload a CSV, validate its columns and show null "
            "and duplicate summaries.",
            "The system shall display request volume by service type.",
            "The system shall calculate and display SLA breach rates by service type.",
            "The system shall display SLA breach rates by queue category to help review workload.",
            "The system shall predict SLA breach probability for new and open requests and "
            "list flagged requests for human review and decision.",
        ],
        "Evidence": [
            "Midterm dataset had 400 rows and 14 columns with missing values and text "
            "inconsistencies that needed cleaning.",
            "Academic Records had the most completed requests (78).",
            "Certification had the highest breach rate (57.89%).",
            "The High queue category had the highest breach rate (70.18%).",
            "Decision Tree reached accuracy 0.82 and breach recall 0.81 on the midterm test set.",
        ],
        "Implemented in": ["Dataset", "Data Analysis", "Data Analysis", "Data Analysis", "Prediction, Dashboard"],
    }))

    data = get_dataset()
    if data is not None and not data["model_df"].empty:
        m = data["model_df"]
        vol = m["Service_Type"].value_counts()
        rates = m.groupby("Service_Type")["SLA_Breached"].apply(lambda s: s.astype(float).mean())
        q = m.groupby("Queue_Category")["SLA_Breached"].apply(lambda s: s.astype(float).mean())
        with st.expander(f"Same evidence computed from the loaded dataset ({data['name']})"):
            st.write(f"- Most completed requests: **{vol.index[0]}** ({int(vol.iloc[0])})")
            st.write(f"- Highest breach rate by service: **{rates.idxmax()}** ({rates.max() * 100:.2f}%)")
            st.write(f"- Highest breach rate by queue category: **{q.idxmax()}** ({q.max() * 100:.2f}%)")

    st.subheader("Non-functional requirements")
    st.table(pd.DataFrame({
        "ID": ["NFR1", "NFR2", "NFR3"],
        "Category": ["Performance", "Security and privacy", "Transparency and accountability"],
        "Requirement": [
            "The trained model shall be loaded once and cached; it shall not be retrained "
            "on page refresh. Cleaning of an uploaded dataset shall also be cached.",
            "Only synthetic or authorized data shall be used; passwords and real student "
            "records shall never be committed to the repository.",
            "Every prediction shall show the model version, a timestamp, limitations and a "
            "human-review notice; every final decision shall be recorded with the reviewer name.",
        ],
    }))

    st.subheader("Updated use case diagram")
    _show_image("use_case_diagram.png", "Use cases and actors matching the implemented pages")

    st.subheader("Analytical process flow and decision workflow")
    _show_image("decision_workflow.png", "From data upload to the final human decision")
    st.markdown(
        "**Decision rule:** the model assigns a risk level (High, Medium, Low) from the "
        "breach probability. Staff review High and Medium requests first, then agree with "
        "the recommended follow-up or override it with notes. The final decision is always "
        "made and recorded by a person."
    )

    with st.expander("Known limitations"):
        st.write("- Small dataset and a shallow Decision Tree; probabilities are coarse.")
        st.write("- The model only learns from completed requests (survivorship bias).")
        st.write("- Decisions and prediction history are stored for the session only.")
        st.write("- No user authentication in this prototype.")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Module", PAGES)

_data = get_dataset()
if _data is not None:
    st.sidebar.success(f"Dataset: {_data['name']}\n\n{len(_data['clean']):,} rows")
else:
    st.sidebar.warning("No dataset loaded")
st.sidebar.caption("Decision support only - human review required.")

if menu == "Home":
    render_home()
elif menu == "Dataset":
    render_dataset_page()
elif menu == "Data Analysis":
    render_eda_page()
elif menu == "Prediction":
    render_prediction_page()
elif menu == "Dashboard":
    render_dashboard_page()
else:
    render_system_design()
