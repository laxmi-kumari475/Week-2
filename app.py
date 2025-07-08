import pandas as pd
import numpy as np
import joblib
import pickle 
import streamlit as st

# ---  Basic Page Configuration for Title, Icon, and Layout ---
st.set_page_config(
    page_title="Water Pollutants Predictor", # Sets the title in the browser tab
    page_icon="💧", # Adds an emoji icon to the browser tab
    layout="centered" # Controls the app's content width (can be "wide")
)

# ---  Cache resources for performance. Model loading now wrapped in a function. ---
@st.cache_resource # This decorator ensures the function runs only once per app session
def load_model_resources():
    try:
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        # --- Logic to extract known station IDs for selectbox ---
        known_station_ids = [col.replace('id_', '') for col in model_cols if col.startswith('id_')]
        if not known_station_ids and 'id' in model_cols: # Fallback if 'id' column itself was used
             known_station_ids = ['1', '2', '3', 'S1', 'S2'] # Example if no specific IDs encoded
        return model, model_cols, known_station_ids
    except FileNotFoundError:
        # ---  Robust error handling if model files are missing ---
        st.error("Error: Model files 'pollution_model.pkl' or 'model_columns.pkl' not found. "
                 "Please ensure they are in the same directory as the app.")
        st.stop() # Stops the app execution if critical files are missing

model, model_cols, known_station_ids = load_model_resources() # --- Call the cached function ---

# --- Page Header  ---
st.title("💧 Water Pollutants Predictor") 
st.markdown("""
    <p style='text-align: center; font-size: 18px; color: #34495e;'>
    Predict the levels of key water pollutants based on specific year and station ID.
    </p>
    """, unsafe_allow_html=True) 
st.markdown("---") 

# --- Subheader for input section ---
st.subheader("📊 Input Parameters") 

# --- Use columns for a cleaner layout of inputs ---
col1, col2 = st.columns(2)

with col1: # Input for year now within a column
    # User inputs
    year_input = st.number_input(
        "Enter Year",
        min_value=2000,
        max_value=2100,
        value=2024, #  Default value set to 2024
        help="Input the year for which you want to predict pollutant levels." # ADDED: Help text
    )

with col2: # Input for station ID now within a column
    # --- Conditional logic to use selectbox if known_station_ids exist ---
    if known_station_ids:
        station_id = st.selectbox(
            "Select Station ID",
            options=sorted(known_station_ids), 
            index=0,
            help="Choose the monitoring station ID from the available options." 
        )
    else: # Fallback to text input if no specific IDs were found
        station_id = st.text_input(
            "Enter Station ID",
            value='1',
            help="Input the monitoring station ID. (e.g., '1', 'S-A')" 
        )
        st.info("No specific station IDs found in model_columns.pkl for selection. "
                "Please enter the ID manually as per your training data.") 
st.markdown("---")

# To encode and then predict
if st.button('Predict Pollutant Levels', use_container_width=True): 
    if not station_id:
        st.warning('Please enter or select a Station ID to proceed.') 
    else:
       
        with st.spinner('Calculating predictions...'):
            # Prepare the input
            input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
            #  Added 'prefix' for clarity, though pd.get_dummies defaults to 'id_'
            input_encoded = pd.get_dummies(input_df, columns=['id'], prefix='id')

            # Align with model cols
            aligned_input = pd.DataFrame(0, index=input_encoded.index, columns=model_cols)
            for col in input_encoded.columns:
                if col in model_cols:
                    aligned_input[col] = input_encoded[col]

            # Predict
            try: 
                predicted_pollutants = model.predict(aligned_input)[0] 
                pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL'] 

                st.success("Prediction Complete!")

                # --- Subheader for results ---
                st.subheader(f"Results for Station '{station_id}' in {year_input}:")
               
                res_col1, res_col2 = st.columns(2)
                for i, (p, val) in enumerate(zip(pollutants, predicted_pollutants)):
                    if i % 2 == 0:
                        with res_col1:
                            st.metric(label=f"**{p}**", value=f"{val:.2f}")
                    else:
                        with res_col2:
                            st.metric(label=f"**{p}**", value=f"{val:.2f}")

             

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please verify the inputs and ensure the model is compatible.")



# --- ADDED: Optional 'How it works' expander for more info ---
with st.expander("❓ How does this work?"):
    st.write("""
        This application uses a pre-trained machine learning model to estimate water pollutant levels.
        You provide a 'Year' and a 'Station ID', and the model then predicts the concentrations of
        various pollutants like Oxygen, Nitrates, Sulfates, etc.

        The model was trained on historical water quality data.
        Accuracy depends on the quality and characteristics of the data the model was trained on.
    """)
