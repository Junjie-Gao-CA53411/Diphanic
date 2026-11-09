import streamlit as st
import os
import shutil
import yaml
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"

SUPPORTED_EXTENSIONS = [".yaml", ".yml", ".json", ".jsonl", ".csv", ".py"]


# Streamlit UI

st.set_page_config(
    page_title="LabPanel",
    page_icon= BASE / "logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LabPanel - Tritlab Experiments Dashboard")

# Sidebar = Load Experiments and Configurations

# Read experiment files
def discover_experiments(root: Path):
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())

def list_experiment_files(exp_dir: Path):
    return sorted(f for f in exp_dir.iterdir() if f.suffix in SUPPORTED_EXTENSIONS)

experiments = discover_experiments(EXPERIMENTS_ROOT)

with st.sidebar:
    st.title("Experiment Configuration")

    if not experiments:
        st.warning("No experiments found in the experiments directory.")
    else:
        experiment_names = [exp.name for exp in experiments]
        selected_experiment = st.selectbox("Select an Experiment", experiment_names)
        selected_experiment_dir = EXPERIMENTS_ROOT / selected_experiment

if selected_experiment:
        config = load_experiment_config(selected_experiment_dir)

        with st.sidebar:
            st.divider()
            st.subheader(selected_experiment_dir.name)

            name = st.text_input("Experiment Name", value=config.get("experiment", selected_experiment_dir.name))
            

