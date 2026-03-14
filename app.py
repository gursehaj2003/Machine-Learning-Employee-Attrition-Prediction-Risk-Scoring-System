import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    le_dict = joblib.load('label_encoders.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('attrition_model.pkl')
    num_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'HourlyRate', 'JobLevel', 
                'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    return le_dict, preprocessor, model, num_cols, cat_cols

le_dict, preprocessor, model, num_cols, cat_cols = load_model()

st.title("🔄 Employee Attrition Prediction System")

with st.form(key="prediction_form"):
    # Categorical inputs
    business_travel = st.selectbox("BusinessTravel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    department = st.selectbox("Department", ['Human Resources', 'Research & Development', 'Sales'])
    education_field = st.selectbox("EducationField", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    job_role = st.selectbox("JobRole", ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                                        'Healthcare Representative', 'Manager', 'Sales Executive', 'Research Director', 
                                        'Sales Representative', 'Human Resources', 'Sales Manager'])
    marital_status = st.selectbox("MaritalStatus", ['Single', 'Married', 'Divorced'])
    overtime = st.selectbox("OverTime", ['Yes', 'No'])
    
    # Numerical inputs (with realistic defaults)
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    daily_rate = st.number_input("DailyRate", min_value=100, max_value=2000, value=800)
    distance_from_home = st.number_input("DistanceFromHome", min_value=1, max_value=30, value=5)
    education = st.number_input("Education", min_value=1, max_value=5, value=3)
    hourly_rate = st.number_input("HourlyRate", min_value=30, max_value=100, value=65)
    job_level = st.number_input("JobLevel", min_value=1, max_value=5, value=2)
    monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=20000, value=5000)
    num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, max_value=9, value=2)
    percent_salary_hike = st.number_input("PercentSalaryHike", min_value=10, max_value=25, value=15)
    performance_rating = st.number_input("PerformanceRating", min_value=1, max_value=4, value=3)
    stock_option_level = st.number_input("StockOptionLevel", min_value=0, max_value=3, value=1)
    total_working_years = st.number_input("TotalWorkingYears", min_value=0, max_value=40, value=5)
    training_times_last_year = st.number_input("TrainingTimesLastYear", min_value=0, max_value=7, value=2)
    work_life_balance = st.number_input("WorkLifeBalance", min_value=1, max_value=4, value=3)
    years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)
    years_in_current_role = st.number_input("YearsInCurrentRole", min_value=0, max_value=20, value=2)
    years_since_last_promotion = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=15, value=1)
    years_with_curr_manager = st.number_input("YearsWithCurrManager", min_value=0, max_value=15, value=2)
    
    submit_button = st.form_submit_button(label="🚀 Predict Attrition Risk", use_container_width=True)

if submit_button:
    # Prepare inputs dict
    inputs = {
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'Gender': gender,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'OverTime': overtime,
        'Age': age, 'DailyRate': daily_rate, 'DistanceFromHome': distance_from_home,
        'Education': education, 'HourlyRate': hourly_rate, 'JobLevel': job_level,
        'MonthlyIncome': monthly_income, 'NumCompaniesWorked': num_companies_worked,
        'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
        'StockOptionLevel': stock_option_level, 'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year, 'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion, 'YearsWithCurrManager': years_with_curr_manager
    }
    
    # Feature engineering
    inputs['IncomeExperienceRatio'] = inputs['MonthlyIncome'] / (inputs['TotalWorkingYears'] + 1)
    inputs['PromotionDelay'] = 1 if inputs['YearsSinceLastPromotion'] > 3 else 0
    inputs['EngagementScore'] = 3.0  # Average; extend with satisfaction fields if dataset includes them
    inputs['WorkloadStress'] = 1 if inputs['OverTime'] == 'Yes' else 0
    
    # Create DataFrame with exact order
    feature_order = cat_cols + num_cols + ['IncomeExperienceRatio', 'PromotionDelay', 'EngagementScore', 'WorkloadStress']
    features = pd.DataFrame([inputs])[feature_order]
    
    # Encode categoricals
    for col in cat_cols:
        features[col] = le_dict[col].transform(features[col])
    
    # Preprocess
    X_pre = preprocessor.transform(features)
    
    # Predict
    prob = model.predict_proba(X_pre)[0][1]
    risk_level = "🟥 High Risk (>50%)" if prob > 0.5 else "🟢 Low Risk (≤50%)"
    
    st.success(f"**Attrition Probability: {prob:.2%}**")
    st.balloons()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Level", risk_level)
    with col2:
        st.metric("Recommendation", "Target Retention" if prob > 0.5 else "Monitor")
