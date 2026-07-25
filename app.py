import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Genomic Triage Tool", page_icon="🧬", layout="wide")

# --- Custom CSS for Medical SaaS UI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #0056b3; color: white; border-radius: 5px; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 15px;
    }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
    st.stop()

model = load_model(MODEL_PATH)

# --- Header ---
st.title("🧬 Genomic Breast Cancer Triage Tool")
st.markdown("Upload a patient's gene expression profile (TCGA format .txt or .csv) to receive instant malignancy prediction and biomarker analysis.")
st.markdown("---")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Patient Gene Expression Data", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # Read data based on file type
        if uploaded_file.name.endswith('.txt'):
            # TCGA txt files are tab-separated and have genes as rows
            raw_df = pd.read_csv(uploaded_file, sep="\t")
            # Check if it needs transposition (if 'Hybridization REF' is a column)
            if "Hybridization REF" in raw_df.columns:
                input_df = raw_df.set_index("Hybridization REF").T
            else:
                input_df = raw_df
        else:
            input_df = pd.read_csv(uploaded_file)
        
        # Drop label column if it accidentally exists in the uploaded data
        if "label" in input_df.columns:
            input_df = input_df.drop("label", axis=1)
            
        # Basic validation
        if input_df.shape[1] < 1000:
            st.warning("Warning: The uploaded file has significantly fewer features than the expected 17,814 genes. Predictions may fail or be inaccurate.")
        
        st.success(f"File processed successfully. Detected {input_df.shape[0]} sample(s) and {input_df.shape[1]} gene features.")
        
        # Predict
        with st.spinner("Analyzing genomic profile..."):
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)[:, 1] # Probability of Tumor (class 1)
            
            # Display results for each sample
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                st.subheader(f"Sample {i+1} Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    if pred == 1:
                        st.error(f"Prediction: Tumor")
                        st.metric(label="Malignancy Confidence", value=f"{prob*100:.2f}%")
                    else:
                        st.success(f"Prediction: Normal")
                        st.metric(label="Benign Confidence", value=f"{(1-prob)*100:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.write("### Top 5 Biomarker Drivers")
                    st.write("*Genes with highest ANOVA F-score contributing to this prediction*")
                    
                    # Extract feature selection data from the pipeline
                    selector = model.named_steps['selector']
                    selected_mask = selector.get_support()
                    
                    # Ensure we only look at genes that were in the input and selected
                    # The selector was fit on training data, so it has a fixed set of features
                    # If input_df columns match training columns, this works perfectly
                    if len(selected_mask) == input_df.shape[1]:
                        scores = selector.scores_[selected_mask]
                        gene_names = input_df.columns[selected_mask]
                        
                        gene_df = pd.DataFrame({
                            'Gene': gene_names, 
                            'ANOVA F-score': scores
                        }).sort_values(by='ANOVA F-score', ascending=False).head(5)
                        
                        st.dataframe(gene_df.style.format({"ANOVA F-score": "{:.2f}"}), use_container_width=True, hide_index=True)
                    else:
                        st.warning("Feature mismatch: The uploaded file's gene count does not match the training data exactly. Biomarker extraction skipped.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                st.markdown("---")
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please ensure the file format matches the TCGA training data (tab-separated, genes as rows, samples as columns).")
else:
    st.info("Awaiting file upload...")