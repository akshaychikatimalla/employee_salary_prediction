import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Load Model Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        train_cols = joblib.load('train_columns.pkl')
        categorical_options_df = joblib.load('categorical_options_df.pkl')

        original_categorical_cols = ['JobRole', 'Location', 'TechStack', 'Department', 'Gender', 'EmploymentType']

        return model, scaler, train_cols, categorical_options_df, original_categorical_cols
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}")
        st.error("Please ensure 'best_rf_model.pkl', 'scaler.pkl', 'train_columns.pkl', and 'categorical_options_df.pkl' are in the same directory as this Streamlit app.")
        st.stop()

# Load all necessary artifacts
model, scaler, train_cols, categorical_options_df, original_categorical_cols = load_artifacts()

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Employee Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced UI and Animations ---
st.markdown("""
<style>
    /* General Body Styling */
    body {
        font-family: 'Segoe UI', sans-serif;
        color: #34495E; /* Dark Grey */
        background-color: #F8F9F9; /* Light off-white */
    }

    /* Hero Section - Top Banner */
    .hero-section {
        background-image: url('https://www.mrc-asia.com/img/predictive_banner.41d2f5e7.jpg')
        background-size: cover;
        background-position: center;
        padding: 80px 20px;
        color: white;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 40px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5); /* Dark overlay */
        border-radius: 10px;
    }
    .hero-content {
        position: relative;
        z-index: 1;
    }
    .hero-title {
        font-size: 4.5em; /* Even larger font */
        font-weight: bold;
        margin-bottom: 15px;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7);
        animation: scaleIn 1.5s ease-out forwards;
        opacity: 0; /* Start hidden for animation */
    }
    .hero-subtitle {
        font-size: 1.8em;
        line-height: 1.5;
        max-width: 800px;
        margin: 0 auto;
        animation: fadeInText 2s ease-in-out 1s forwards; /* Delayed fade in */
        opacity: 0; /* Start hidden for animation */
    }

    /* Section Headers */
    .st-emotion-cache-nahz7x h2 { /* Targeting Streamlit's generated h2 for subheaders */
        color: #2E86C1; /* A nice blue */
        border-bottom: 3px solid #D6EAF8; /* Thicker light blue underline */
        padding-bottom: 12px;
        margin-bottom: 25px;
        font-size: 1.8em;
        font-weight: 600;
        text-align: center; /* Center align subheaders */
        animation: slideInUp 1s ease-out;
    }

    /* Custom Dividers */
    .divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 50px 0;
        color: #5D6D7E;
        font-size: 1.2em;
        font-weight: bold;
    }
    .divider::before,
    .divider::after {
        content: '';
        flex: 1;
        border-bottom: 2px dashed #D5DBDB;
    }
    .divider:not(:empty)::before {
        margin-right: .8em;
    }
    .divider:not(:empty)::after {
        margin-left: .8em;
    }
    .divider-icon {
        font-size: 1.5em;
        color: #28B463; /* Green icon */
        margin: 0 10px;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #28B463; /* Green for predict button */
        color: white;
        font-weight: bold;
        border-radius: 12px; /* More rounded */
        padding: 12px 25px;
        border: none;
        transition: all 0.3s ease; /* Smooth transition for all properties */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #239B56; /* Darker green on hover */
        transform: translateY(-3px); /* Lift effect on hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        cursor: pointer;
    }

    /* Success Message Styling with Animation */
    .predicted-salary-box {
        background-color: #D4EDDA; /* Light green background */
        color: #155724; /* Dark green text */
        border: 2px solid #C3E6CB; /* Green border */
        padding: 25px;
        border-radius: 15px;
        font-size: 2.2em; /* Larger font */
        text-align: center;
        margin-top: 30px;
        font-weight: bold;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
        animation: popIn 0.7s ease-out; /* Pop-in animation */
    }
    .predicted-salary-box strong {
        color: #0F4229; /* Even darker green for emphasis */
    }

    /* General text styling */
    p, li {
        color: #34495E; /* Dark grey for general text */
        line-height: 1.6;
    }

    /* Input Widget Styling (subtle) */
    .stSlider > div > div > div {
        background-color: #EBF5FB; /* Light blue background for slider track */
    }
    .stSelectbox > div > div {
        border-radius: 8px; /* Rounded select boxes */
    }


    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInText {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.7); }
        to { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown(f"""
<div class="hero-section">
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <div class="hero-title"> Employee Salary Predictor </div>
        <div class="hero-subtitle">
            Uncover the earning potential of any role. Our advanced AI model provides accurate salary estimations
            based on key professional and personal attributes.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Section for Input Fields ---
st.header("Enter Employee Details")
st.markdown("""
    <p style="text-align: center; margin-bottom: 30px; font-size: 1.1em;">
        Adjust the parameters below to see how different factors influence salary predictions.
    </p>
""", unsafe_allow_html=True)

# Store initial default values for validation (as discussed)
default_values = {
    'years_experience': 5,
    'education_level': 1,
    'age': 30,
    'certifications': 1,
    'previous_companies': 1,
    'performance_rating': 3,
    'working_hours': 40,
    'leaves_taken': 10,
    'remote_work': 0
}

# Organize input fields into three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("General Information ℹ️")
    years_experience = st.slider("Years Experience", 0, 30, default_values['years_experience'], key='years_experience_slider', help="Number of years in current or similar roles.")
    education_level = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=default_values['education_level'] - 1, format_func=lambda x: f"Level {x}", key='education_level_select', help="1: High School, 2: Associates, 3: Bachelor's, 4: Master's, 5: PhD.")
    age = st.slider("Age", 20, 65, default_values['age'], key='age_slider', help="Employee's current age.")
    certifications = st.slider("Certifications", 0, 5, default_values['certifications'], key='certifications_slider', help="Number of relevant professional certifications.")

with col2:
    st.subheader("Role & Location 🏢")
    job_role = st.selectbox("Job Role", categorical_options_df['JobRole'].unique().tolist(), key='job_role_select', help="Specific job title or function.")
    location = st.selectbox("Location", categorical_options_df['Location'].unique().tolist(), key='location_select', help="Geographic location of the employment.")
    department = st.selectbox("Department", categorical_options_df['Department'].unique().tolist(), key='department_select', help="The department the employee belongs to.")
    tech_stack = st.selectbox("Tech Stack", categorical_options_df['TechStack'].unique().tolist(), key='tech_stack_select', help="Primary technology stack used (e.g., Python, Java, JavaScript).")

with col3:
    st.subheader("Work & Performance 📊")
    previous_companies = st.slider("Previous Companies", 0, 5, default_values['previous_companies'], key='previous_companies_slider', help="Number of companies worked at previously.")
    performance_rating = st.slider("Performance Rating", 1, 5, default_values['performance_rating'], key='performance_rating_slider', help="Employee's last performance review rating (1-5, 5 being best).")
    working_hours = st.slider("Working Hours (per week)", 35, 60, default_values['working_hours'], key='working_hours_slider', help="Average weekly working hours.")
    leaves_taken = st.slider("Leaves Taken (per year)", 0, 30, default_values['leaves_taken'], key='leaves_taken_slider', help="Total number of leaves taken in a year.")
    gender = st.selectbox("Gender", categorical_options_df['Gender'].unique().tolist(), key='gender_select', help="Employee's gender.")
    employment_type = st.selectbox("Employment Type", categorical_options_df['EmploymentType'].unique().tolist(), key='employment_type_select', help="Full-time, Part-time, Contract, etc.")
    remote_work = st.selectbox("Remote Work", [0, 1], index=default_values['remote_work'], format_func=lambda x: "Yes" if x == 1 else "No", key='remote_work_select', help="Is the position remote? (1 for Yes, 0 for No).")

# --- Divider before Prediction ---
st.markdown("""
<div class="divider">
    <span class="divider-icon">✨</span> Predict Now <span class="divider-icon">✨</span>
</div>
""", unsafe_allow_html=True)

# --- Prediction Button and Logic ---
if st.button("Calculate Salary Prediction 🚀", help="Click to get the estimated salary based on the entered details."):
    with st.spinner("Crunching numbers..."):
        # 1. Create a DataFrame from user inputs
        input_data = pd.DataFrame([{
            'YearsExperience': years_experience,
            'EducationLevel': education_level,
            'Age': age,
            'JobRole': job_role,
            'Location': location,
            'Certifications': certifications,
            'PreviousCompanies': previous_companies,
            'PerformanceRating': performance_rating,
            'TechStack': tech_stack,
            'Department': department,
            'WorkingHours': working_hours,
            'LeavesTaken': leaves_taken,
            'Gender': gender,
            'EmploymentType': employment_type,
            'RemoteWork': remote_work
        }])

        # 2. Apply Feature Engineering (must be consistent with training notebook)
        # Note: Added a small constant to Age to prevent division by zero for Experience_Age_Ratio
        input_data['Experience_Age_Ratio'] = input_data['YearsExperience'] / (input_data['Age'] + 1e-6)
        input_data['Experience_Age_Ratio'].fillna(0, inplace=True)

        input_data['Total_Skills_Certifications'] = input_data['Certifications'] + input_data['EducationLevel']

        # Add a small constant to LeavesTaken to prevent division by zero, as done in notebook
        input_data['Productivity_Score'] = input_data['PerformanceRating'] * (input_data['WorkingHours'] / (input_data['LeavesTaken'] + 1e-6))
        input_data['Productivity_Score'].fillna(0, inplace=True)

        # 3. Apply One-Hot Encoding (must be consistent with training notebook)
        # Create a dummy DataFrame with all training columns to ensure consistent encoding
        # This is crucial for new inputs that might not contain all categories seen during training
        dummy_row_for_alignment = pd.DataFrame(0, index=[0], columns=train_cols)

        # Concatenate input data with the dummy row.
        # This helps `pd.get_dummies` create all expected columns,
        # even if an input doesn't have a specific category.
        combined_for_encoding_inference = pd.concat([dummy_row_for_alignment, input_data], ignore_index=True)

        # Ensure categorical columns are 'category' dtype for get_dummies consistency
        for col in original_categorical_cols:
            if col in combined_for_encoding_inference.columns:
                combined_for_encoding_inference[col] = combined_for_encoding_inference[col].astype('category')

        # Apply one-hot encoding
        encoded_input_full = pd.get_dummies(combined_for_encoding_inference, columns=original_categorical_cols, drop_first=True)

        # Extract only the newly encoded input row (it's the second row after concatenation)
        # And reindex to ensure the exact same column order as X_train
        X_input_aligned = encoded_input_full.iloc[1].reindex(train_cols, fill_value=0.0).to_frame().T

        # 4. Apply Scaling (must be consistent with training notebook)
        X_input_scaled = scaler.transform(X_input_aligned)

        # 5. Make Prediction
        predicted_salary = model.predict(X_input_scaled)[0]

        st.markdown(f"""
            <div class="predicted-salary-box">
                Estimated Salary: <strong>${predicted_salary:,.2f}</strong> 
                
            </div>
        """, unsafe_allow_html=True)

