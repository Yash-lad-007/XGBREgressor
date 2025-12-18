import streamlit as st
import pandas as pd
import numpy as np
import pickle
import  os
  
# Define file paths for the model and the data
MODEL_FILE = "XGBRegressor.pkl"
DATA_FILE = "Concrete_Data_Yeh.csv"   

# Set up the Streamlit page configuration 
st.set_page_config(
    page_title="Concrete Strength Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the model (uses st.cache_resource for efficient caching)
@st.cache_resource
def load_model(file_path):
    """Loads the pickled XGBoost model."""
    try:
        if not os.path.exists(file_path):
             st.error(f"Model file '{file_path}' not found. Please ensure it is uploaded.")
             return None
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to load the data (uses st.cache_data for efficient caching)
@st.cache_data
def load_data(file_path):
    """Loads the concrete data CSV."""
    try:
        if not os.path.exists(file_path):
             st.error(f"Data file '{file_path}' not found. Please ensure it is uploaded.")
             return None
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None

# Load resources
model = load_model(MODEL_FILE)
concrete_data = load_data(DATA_FILE)

# --- UI LAYOUT & TABS ---

st.title("🔬 Concrete Compressive Strength Predictor")
st.markdown("""
This application uses a pre-trained **XGBoost Regressor** model to estimate the 28-day 
compressive strength (MPa) of concrete. Use the Data Explorer to see the training data context.
""")

tab_predict, tab_data = st.tabs(["Prediction Tool", "Data Explorer"])


# ==============================================================================
# TAB 1: PREDICTION TOOL
# ==============================================================================
with tab_predict:
    if model is None:
        st.warning("Prediction service is unavailable because the model could not be loaded.")
    else:
        # --- Sidebar Inputs ---
        st.sidebar.header("Mix Parameters (Inputs)")
        st.sidebar.markdown("Use the sliders to define the material composition and age.")

        # Define input widgets for the 8 features
        input_data = {}

        # Mass of materials (kg/m³)
        input_data['cement'] = st.sidebar.slider('Cement (kg/m³)', 100.0, 550.0, 300.0, 0.1)
        input_data['slag'] = st.sidebar.slider('Blast Furnace Slag (kg/m³)', 0.0, 350.0, 50.0, 0.1)
        input_data['flyash'] = st.sidebar.slider('Fly Ash (kg/m³)', 0.0, 250.0, 50.0, 0.1)
        input_data['water'] = st.sidebar.slider('Water (kg/m³)', 120.0, 250.0, 180.0, 0.1)
        input_data['superplasticizer'] = st.sidebar.slider('Superplasticizer (kg/m³)', 0.0, 30.0, 5.0, 0.1)
        input_data['coarseaggregate'] = st.sidebar.slider('Coarse Aggregate (kg/m³)', 800.0, 1150.0, 1000.0, 0.1)
        input_data['fineaggregate'] = st.sidebar.slider('Fine Aggregate (kg/m³)', 550.0, 1000.0, 750.0, 0.1)

        # Age (days)
        input_data['age'] = st.sidebar.slider('Age (Days)', 1, 365, 28, 1)


        # --- DATA PREPARATION & PREDICTION ---

        # The order of features MUST match the training data
        feature_list = [
            input_data['cement'],
            input_data['slag'],
            input_data['flyash'],
            input_data['water'],
            input_data['superplasticizer'],
            input_data['coarseaggregate'],
            input_data['fineaggregate'],
            input_data['age']
        ]

        features_array = np.array([feature_list])

        # --- PREDICT BUTTON AND RESULT DISPLAY ---

        st.markdown("---")

        if st.button("Calculate Predicted Strength", use_container_width=True):
            try:
                # Make prediction
                prediction = model.predict(features_array)[0]

                st.subheader("Predicted Compressive Strength")
                
                # Display the result prominently using HTML/CSS for style
                st.markdown(f"""
                <div style="
                    text-align: center;
                    background-color: #d1f7e8; 
                    padding: 30px;
                    border-radius: 12px;
                    border: 3px solid #1abc9c; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                ">
                    <p style="color: #2c3e50; font-size: 1.5em; margin: 0; font-weight: 600;">Prediction (csMPa):</p>
                    <h1 style="color: #1abc9c; font-size: 4em; margin: 10px 0 0 0; font-weight: 900;">{prediction:.2f}</h1>
                    <p style="color: #2c3e50; font-size: 1em; margin: 5px 0 0 0;">MegaPascals</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

        st.markdown("---")

        # Display the input data in the main area
        st.subheader("Current Input Values")
        input_df = pd.DataFrame([input_data])
        input_df.columns = [col.replace('aggregate', ' agg.').capitalize() for col in input_df.columns]
        st.dataframe(input_df.style.format(precision=1), use_container_width=True)

        st.caption("Model details: XGBoost Regressor (100 estimators, max depth 3).")


# ==============================================================================
# TAB 2: DATA EXPLORER
# ==============================================================================
with tab_data:
    if concrete_data is None:
        st.error("Data could not be loaded for the explorer.")
    else:
        st.header("Concrete Mix Dataset (`Concrete_Data_Yeh.csv`)")
        st.markdown("This tab provides insights into the dataset used for training the model, containing **1030 samples**.")

        # --- Statistics ---
        st.subheader("Descriptive Statistics")
        stats_df = concrete_data.describe().T
        st.dataframe(stats_df.style.format(precision=2), use_container_width=True)
        
        # --- Target Distribution ---
        st.subheader("Distribution of Compressive Strength (csMPa)")
        st.bar_chart(concrete_data['csMPa'], color="#1abc9c")
        
        # --- Raw Data ---
        st.subheader("Raw Data Sample")
        st.dataframe(concrete_data.head(20).style.format(precision=2), use_container_width=True)
