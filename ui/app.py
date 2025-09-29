import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st


@st.cache_resource
def load_model():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # change the model path to the correct path
    model_path = os.path.join(project_root, "models", "final_model.pkl")
    if not os.path.exists(model_path):
        st.stop()
    return joblib.load(model_path)


st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title("Heart Disease Prediction")
st.caption("Uses a trained scikit-learn pipeline (imputer + scaler + Logistic Regression)")

model = load_model()

# Auto-process sample_patients.csv and generate results
def process_sample_csv():
    try:
        # Read the sample CSV file
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples(input)", "sample_patients.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            
            # Check if required columns exist
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            if all(col in df.columns for col in required_columns):
                # Select only the required columns
                df_features = df[required_columns]
                
                # Make predictions
                features_array = df_features.values
                predictions = model.predict(features_array)
                
                # Get probabilities if available
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_array)[:, 1]
                else:
                    probabilities = None
                
                # Create results dataframe
                results_df = df_features.copy()
                results_df['Prediction'] = ['Disease' if pred == 1 else 'No Disease' for pred in predictions]
                
                if probabilities is not None:
                    results_df['Probability_Percent'] = (probabilities * 100).round(2)
                    results_df['Risk_Level'] = results_df['Probability_Percent'].apply(
                        lambda x: 'High' if x > 70 else 'Medium' if x > 30 else 'Low'
                    )
                
                # Save results to results folder
                results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
                os.makedirs(results_dir, exist_ok=True)
                
                output_path = os.path.join(results_dir, "heart_disease_predictions.csv")
                results_df.to_csv(output_path, index=False)
                
                return True, output_path, results_df
            else:
                return False, "Missing required columns in sample_patients.csv", None
        else:
            return False, "sample_patients.csv file not found in samples(input) folder", None
    except Exception as e:
        return False, f"Error processing CSV: {str(e)}", None

# Process the sample CSV automatically
success, message, results_df = process_sample_csv()

if success:
    st.success(f"✅ **Auto-processing completed!** Results saved to: `{message}`")
    
    # Display results
    st.subheader("📊 Auto-Generated Prediction Results")
    st.dataframe(results_df)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        disease_count = sum([1 for pred in results_df['Prediction'] if pred == 'Disease'])
        st.metric("Patients with Disease", disease_count)
    
    with col2:
        no_disease_count = len(results_df) - disease_count
        st.metric("Patients without Disease", no_disease_count)
    
    with col3:
        if 'Probability_Percent' in results_df.columns:
            avg_prob_percent = results_df['Probability_Percent'].mean()
            st.metric("Average Risk Probability", f"{avg_prob_percent:.1f}%")
    
    # Download results
    csv_results = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_results,
        file_name="heart_disease_predictions.csv",
        mime="text/csv"
    )
else:
    st.error(f"❌ **Auto-processing failed:** {message}")

st.markdown("---")

# Sample CSV download section
st.subheader("📋 Download Sample CSV Template")

st.write("**Download the current sample_patients.csv file:**")

# Read and display the actual sample_patients.csv file
try:
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples(input)", "sample_patients.csv")
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        
        # Display sample data
        st.write("**Current Sample Data Preview:**")
        st.dataframe(sample_df)
        
        # Download sample CSV
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Current Sample CSV",
            data=sample_csv,
            file_name="sample_patients.csv",
            mime="text/csv",
            help="Download the current sample_patients.csv file"
        )
    else:
        st.error("sample_patients.csv file not found in samples(input) folder!")
except Exception as e:
    st.error(f"Error reading sample file: {str(e)}")

st.markdown("---")
st.subheader("✏️ Manual Input - Single Patient")

# Sample data buttons
st.write("**Quick Test with Sample Data:**")
col_sample1, col_sample2, col_sample3 = st.columns(3)
with col_sample1:
    if st.button("🔴 Sample Patient 1 (High Risk)"):
        st.session_state.sample_data = [65, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
        st.success("High risk patient data loaded!")
with col_sample2:
    if st.button("🟢 Sample Patient 2 (Low Risk)"):
        st.session_state.sample_data = [57, 1, 0, 130, 246, 0, 1, 150, 0, 1.0, 1, 0, 3]
        st.success("Low risk patient data loaded!")
with col_sample3:
    if st.button("🟡 Sample Patient 3 (Medium Risk)"):
        st.session_state.sample_data = [63, 1, 1, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
        st.success("Medium risk patient data loaded!")

col1, col2, col3 = st.columns(3)

# Initialize default values
default_values = [57.0, 1.0, 0.0, 130.0, 246.0, 0.0, 1.0, 150.0, 0.0, 1.0, 1.0, 0.0, 3.0]

# Use sample data if available
if 'sample_data' in st.session_state:
    default_values = st.session_state.sample_data
    # Clear the sample data after using it
    del st.session_state.sample_data

with col1:
    age = st.number_input("age", min_value=18.0, max_value=100.0, value=default_values[0], step=1.0)
    sex = st.selectbox("sex (1=male,0=female)", options=[0.0, 1.0], index=int(default_values[1]))
    cp = st.selectbox("cp (0-3)", options=[0.0, 1.0, 2.0, 3.0], index=int(default_values[2]))
    trestbps = st.number_input("trestbps", min_value=80.0, max_value=220.0, value=default_values[3], step=1.0)
    chol = st.number_input("chol", min_value=100.0, max_value=600.0, value=default_values[4], step=1.0)

with col2:
    fbs = st.selectbox("fbs>120 (1/0)", options=[0.0, 1.0], index=int(default_values[5]))
    restecg = st.selectbox("restecg (0-2)", options=[0.0, 1.0, 2.0], index=int(default_values[6]))
    thalach = st.number_input("thalach", min_value=60.0, max_value=220.0, value=default_values[7], step=1.0)
    exang = st.selectbox("exang (1/0)", options=[0.0, 1.0], index=int(default_values[8]))
    oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=default_values[9], step=0.1)

with col3:
    slope = st.selectbox("slope (0-2)", options=[0.0, 1.0, 2.0], index=int(default_values[10]))
    ca = st.selectbox("ca (0-3)", options=[0.0, 1.0, 2.0, 3.0], index=int(default_values[11]))
    thal = st.selectbox("thal (3=normal,6=fixed,7=reversible)", options=[3.0, 6.0, 7.0], index=[3.0, 6.0, 7.0].index(default_values[12]))

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

if st.button("🔮 Predict"):
    pred = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(features)[0, 1])
    else:
        proba = None

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    label = "Disease" if int(pred) == 1 else "No Disease"
    
    # Display result with color coding
    if int(pred) == 1:
        st.error(f"🔴 **Prediction: {label}**")
    else:
        st.success(f"🟢 **Prediction: {label}**")
        
    if proba is not None:
        proba_percent = proba * 100
        st.write(f"**Probability of disease: {proba_percent:.1f}%**")
        
        # Risk level indicator
        if proba > 0.7:
            st.warning("⚠️ **High Risk**")
        elif proba > 0.3:
            st.info("⚠️ **Medium Risk**")
        else:
            st.success("✅ **Low Risk**")

st.markdown("---")
st.caption("Note: Inputs must match the 13-feature Cleveland format used during training.")


