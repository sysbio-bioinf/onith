import streamlit as st
import json
import os
import sys
from pathlib import Path
from onith.ontology_utils import hpath_term_info_to_json

st.set_page_config(layout="wide")

# UI Styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 0.85em !important;
    }
    </style>
""", unsafe_allow_html=True)

# Caching
@st.cache_data
def load_terms(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_hpath():
    return hpath_term_info_to_json("lesion")

# Argument parsing
if len(sys.argv) < 6:
    st.error("Missing required arguments: JSON path for terms, Hpath info, save dir, sharable flag, and show_term_info flag.")
    st.stop()

json_path_terms = sys.argv[1]
json_path_hpath = sys.argv[2]
save_dir = sys.argv[3]
sharable = sys.argv[4].lower() == "true"
show_term_info = sys.argv[5].lower() == "true"

# Load data
terms_data = []
hpath_data = {}

if sharable:
    uploaded_terms = st.file_uploader("📄 Upload your 'terms_for_manualmapping' JSON file", type="json", key="terms")
    uploaded_hpath = st.file_uploader("📘 Upload your 'hpath_term_info' JSON file", type="json", key="hpath") if show_term_info else None

    if uploaded_terms:
        terms_data = json.load(uploaded_terms)
        if show_term_info:
            if uploaded_hpath:
                hpath_data = json.load(uploaded_hpath)
            else:
                st.warning("Please upload the Hpath term info JSON file.")
                st.stop()
    else:
        st.warning("Please upload the terms JSON file.")
        st.stop()
else:
    terms_data = load_terms(json_path_terms)
    if show_term_info:
        hpath_data = load_hpath()
        # Extract Hpath info
        term_info = hpath_data["term_info"]
        synonym_to_main = hpath_data["synonym_to_main"]

# Prepare terms
row_term_orders = [item["Row Term Order"] for item in terms_data]

if "selections" not in st.session_state:
    st.session_state.selections = [None] * len(terms_data)

# Save/Load
def load_progress():
    path = st.session_state.get("load_path", save_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        st.session_state.selections = data.get("selections", [None] * len(terms_data))
        st.toast("Progress loaded successfully!", icon="✅")
    else:
        st.toast("File not found.", icon="❌")

def save_progress():
    path = st.session_state.get("save_path", save_dir)
    if all(selection is None for selection in st.session_state.selections):
        st.toast("Warning! Cannot save: No terms have been selected yet.", icon="⚠️")
        return
    data = {"selections": st.session_state.selections}
    with open(path, "w") as f:
        json.dump(data, f)
    st.toast("Progress saved successfully!", icon="✅")

# Sidebar
st.sidebar.text_input("Save path", value=save_dir, key="save_path")
st.sidebar.button("💾 Save Progress", on_click=save_progress)

st.sidebar.text_input("Load path", value=save_dir, key="load_path")
st.sidebar.button("🔄 Load Progress", on_click=load_progress)

st.sidebar.write(f"Default progress file: {save_dir}")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ Instructions
- Use the dropdowns to manually map each of your original terms to a standardized ontology term.
- The dropdown menu suggests the most similar terms first, but you can search interactively.
- If the exact term is not available, map it to the next available parent term.
- Leave the dropdown empty to exclude a term from the lesion dictionary (i.e. this term is actually synonym to "no alteration observed" or "normal"; the finding will not be removed from the dataset, but will be mapped to "NORMAL" instead)
- Click **Save Progress** regularly to avoid losing your work.
""")

# Main UI
st.subheader("Manual Term Mapping")

# Dynamically determine column names
example_entry = terms_data[0]
column_keys = [key for key in example_entry.keys() if key not in ["Row Term Order"]]

header_cols = st.columns([3, 4, 9])
for i, key in enumerate(column_keys[:2]):
    with header_cols[i]:
        st.markdown(f"**{key}**")
with header_cols[2]:
    st.markdown("**Mapped Term or Synonym**")

for idx, term in enumerate(terms_data):
    cols = st.columns([3, 4, 9])
    for i, key in enumerate(column_keys[:2]):
        with cols[i]:
            st.markdown(f"{term[key]}")
    with cols[2]:
        c1, c2 = st.columns([10, 1])
        selection = st.session_state.selections[idx]
        with c1:
            selected = st.selectbox(
                "Select term",
                options=[""] + row_term_orders[idx],
                key=f"select_{idx}",
                index=0 if not selection or selection not in row_term_orders[idx] else row_term_orders[idx].index(selection) + 1,
                label_visibility="collapsed"
            )
        st.session_state.selections[idx] = selected if selected else None
        with c2:
            if show_term_info and selected:
                with st.popover("ℹ️"):
                    main_term = synonym_to_main.get(selected, selected)
                    info = term_info.get(main_term)
                    if info:
                        st.markdown(f"""
                        assigned hpath main term:<br>
                        <strong>{info['main_term']}</strong><br>
                        <b>Parent:</b> {info['parent_term']}<br>
                        <b>Definition:</b> {info['definition']}<br>
                        """, unsafe_allow_html=True)
            elif show_term_info:
                st.button("ℹ️", key=f"info_disabled_{idx}", disabled=True)
