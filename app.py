import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Genomic Triage Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Clinical color palette (mirrors .streamlit/config.toml)
# ----------------------------------------------------------------------------
PRIMARY = "#0F4C81"
PRIMARY_DARK = "#0A3660"
PRIMARY_LIGHT = "#E8F0F8"
GREY_BORDER = "#E1E4E8"
TEXT_MUTED = "#5A6472"
DANGER = "#B3261E"
DANGER_BG = "#FBEAE9"
SAFE = "#1E7B4D"
SAFE_BG = "#E9F7EF"

# ----------------------------------------------------------------------------
# Style: ONLY styles elements we create ourselves. Does not touch Streamlit's
# own header, sidebar toggle, or chrome — the .streamlit/config.toml theme
# block handles the overall light/dark backdrop natively and reliably.
# ----------------------------------------------------------------------------
st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .block-container {{padding-top: 5rem; padding-bottom: 2rem; max-width: 1100px;}}
    html, body, [class*="css"] {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }}

    .stButton>button, .stDownloadButton>button {{
        background-color: {PRIMARY};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background-color: {PRIMARY_DARK};
        color: white;
    }}

    .metric-card {{
        background-color: white;
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid {GREY_BORDER};
        box-shadow: 0 2px 8px rgba(15,76,129,0.06);
        margin-bottom: 15px;
        height: 100%;
    }}

    .hero-icon {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        width: 72px; height: 72px;
        border-radius: 18px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.1rem;
        box-shadow: 0 4px 10px rgba(15,76,129,0.25);
        margin-top: 4px;
    }}
    .hero-title {{
        color: {PRIMARY_DARK};
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0;
        line-height: 1.2;
    }}
    .hero-subtitle {{
        color: {TEXT_MUTED};
        font-size: 1.0rem;
        margin-top: 4px;
    }}

    .disclaimer-banner {{
        background-color: {PRIMARY_LIGHT};
        border-left: 4px solid {PRIMARY};
        padding: 12px 16px;
        border-radius: 6px;
        color: {PRIMARY_DARK};
        font-size: 0.88rem;
        margin: 18px 0 24px 0;
    }}

    .result-banner {{
        border-radius: 10px;
        padding: 14px 18px;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }}
    .result-tumor {{ background-color: {DANGER_BG}; color: {DANGER}; border: 1px solid rgba(179,38,30,0.25); }}
    .result-normal {{ background-color: {SAFE_BG}; color: {SAFE}; border: 1px solid rgba(30,123,77,0.25); }}

    .empty-state {{
        background-color: white;
        border: 1.5px dashed {GREY_BORDER};
        border-radius: 14px;
        padding: 64px 30px;
        text-align: center;
        color: {TEXT_MUTED};
        margin-top: 24px;
    }}
    .empty-state .icon {{ font-size: 2.8rem; margin-bottom: 14px; }}
    .empty-state .title {{ color: {PRIMARY_DARK}; font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; }}

    .sidebar-footer {{
        font-size: 0.72rem;
        color: {TEXT_MUTED};
        margin-top: 24px;
        line-height: 1.4;
    }}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model Loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
    st.stop()

model = load_model(MODEL_PATH)

# ----------------------------------------------------------------------------
# Sidebar: upload + documentation
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 Triage Console")
    st.caption("Genomic classification workspace")
    st.markdown("---")

    st.markdown("#### Upload patient data")
    uploaded_file = st.file_uploader(
        "Gene expression profile",
        type=["csv", "txt"],
        help="TCGA-format .txt (tab-separated, genes as rows) or .csv (samples as rows, genes as columns).",
        label_visibility="collapsed",
    )
    st.caption("Accepted formats: TCGA `.txt` or `.csv`")

    st.markdown("---")

    with st.expander("ℹ️ About this tool"):
        st.markdown(
            "This tool classifies breast tissue samples as **Tumor** or **Normal** "
            "from RNA-Seq gene expression profiles sourced from **The Cancer Genome "
            "Atlas (TCGA)**, a public, NIH-funded genomics program spanning 33 cancer "
            "types.\n\n"
            "- **Input space:** 17,814 gene expression features\n"
            "- **Model:** scikit-learn pipeline with ANOVA F-score feature selection\n"
            "- **Output:** malignancy probability + top contributing biomarkers"
        )

    with st.expander("⚙️ Model details"):
        st.markdown(
            "- Pipeline: `SelectKBest` (ANOVA F-test) → classifier\n"
            "- Trained on paired tumor / normal TCGA-BRCA expression data\n"
            "- Biomarkers ranked by ANOVA F-score computed on the training set"
        )

    st.markdown(
        "<div class='sidebar-footer'>⚠️ Research &amp; educational use only.<br>"
        "Not a clinical diagnostic device.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Hero section
# ----------------------------------------------------------------------------
col_icon, col_title = st.columns([1, 8])
with col_icon:
    st.markdown("<div class='hero-icon'>🧬</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<p class='hero-title'>Genomic Breast Cancer Triage Tool</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-subtitle'>Upload a gene expression profile to receive an instant "
        "malignancy assessment and biomarker breakdown.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='disclaimer-banner'>⚠️ <b>Research &amp; educational tool only.</b> "
    "Predictions are generated by a machine learning model trained on public TCGA data "
    "and are not a substitute for professional pathological diagnosis.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Main content: results or empty state
# ----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.txt'):
            raw_df = pd.read_csv(uploaded_file, sep="\t")
            if "Hybridization REF" in raw_df.columns:
                input_df = raw_df.set_index("Hybridization REF").T
            else:
                input_df = raw_df
        else:
            input_df = pd.read_csv(uploaded_file)

        if "label" in input_df.columns:
            input_df = input_df.drop("label", axis=1)

        if input_df.shape[1] < 1000:
            st.warning(
                "Warning: The uploaded file has significantly fewer features than the "
                "expected 17,814 genes. Predictions may fail or be inaccurate."
            )

        st.success(
            f"File processed successfully. Detected {input_df.shape[0]} sample(s) "
            f"and {input_df.shape[1]} gene features."
        )

        report_lines = [
            "Genomic Breast Cancer Triage Tool — Summary Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Source file: {uploaded_file.name}",
            "For research and educational use only. Not a clinical diagnosis.",
            "-" * 60,
        ]

        with st.spinner("Analyzing genomic profile..."):
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)[:, 1]

            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                st.markdown(f"### Sample {i + 1}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    if pred == 1:
                        confidence = prob
                        st.markdown(
                            "<div class='result-banner result-tumor'>🔴 Prediction: Tumor</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric(label="Malignancy Confidence", value=f"{confidence * 100:.2f}%")
                        report_lines.append(f"Sample {i + 1}: TUMOR (confidence {confidence * 100:.2f}%)")
                    else:
                        confidence = 1 - prob
                        st.markdown(
                            "<div class='result-banner result-normal'>🟢 Prediction: Normal</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric(label="Benign Confidence", value=f"{confidence * 100:.2f}%")
                        report_lines.append(f"Sample {i + 1}: NORMAL (confidence {confidence * 100:.2f}%)")
                    st.progress(min(max(confidence, 0.0), 1.0))
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.write("**Top 5 Biomarker Drivers**")
                    st.caption("Genes with highest ANOVA F-score contributing to this prediction")

                    selector = model.named_steps['selector']
                    selected_mask = selector.get_support()

                    if len(selected_mask) == input_df.shape[1]:
                        scores = selector.scores_[selected_mask]
                        gene_names = input_df.columns[selected_mask]

                        gene_df = pd.DataFrame({
                            'Gene': gene_names,
                            'ANOVA F-score': scores
                        }).sort_values(by='ANOVA F-score', ascending=False).head(5)

                        st.dataframe(
                            gene_df.style.format({"ANOVA F-score": "{:.2f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                        report_lines.append(
                            "  Top biomarkers: "
                            + ", ".join(gene_df['Gene'].astype(str).tolist())
                        )
                    else:
                        st.warning(
                            "Feature mismatch: the uploaded file's gene count does not "
                            "match the training data exactly. Biomarker extraction skipped."
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

        report_lines.append("-" * 60)
        st.download_button(
            label="⬇ Download Summary Report",
            data="\n".join(report_lines),
            file_name=f"triage_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info(
            "Please ensure the file format matches the TCGA training data "
            "(tab-separated, genes as rows, samples as columns)."
        )
else:
    st.markdown(
        "<div class='empty-state'>"
        "<div class='icon'>🧬</div>"
        "<div class='title'>Awaiting patient genomic profile</div>"
        "<div>Upload a TCGA-format file from the sidebar to begin analysis.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
