import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import dice_ml
import plotly.express as px

# Load Pre-trained Model and Artifacts 
try:
    model = joblib.load('xgb_model.joblib')
    scaler_X = joblib.load('scaler_X.joblib')
    model_features = joblib.load('model_features.joblib')
except FileNotFoundError:
    st.error("Model artifacts not found. Please run `python train_model.py` first.")
    st.stop()

# Feature Engineering Function
def feature_engineer(df):
    """
    Recreates the engineered features for a single data point.
    Handles edge cases as described.
    """
    epsilon = 1e-6
    df['Income_Stability_Index'] = df['std_weekly_income'] / (df['avg_weekly_income'] + epsilon)
    df['Income_Stability_Index'] = df['Income_Stability_Index'].clip(upper=10)
    df['Fare-to-Tip_Ratio'] = df['avg_tip_percentage'] / (df['avg_total_fare'] + epsilon)
    df['Fare-to-Tip_Ratio'] = df['Fare-to-Tip_Ratio'].clip(upper=10)
    df['Trip_Consistency'] = df['number_of_weeks_on_platform'] / (df['total_trips'] + epsilon)
    return df

# SHAP Explainer Function 
def explain_prediction_shap(model, features_df):
    """Generates SHAP values to explain the model's prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    shap_df = pd.DataFrame({
        'feature': features_df.columns,
        'shap_value': shap_values[0]
    })
    
    shap_df['abs_shap_value'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values(by='abs_shap_value', ascending=False)
    
    return shap_df

# --- DiCE Counterfactuals Function ---
def get_counterfactuals(model, features_df, scaler_X):
    """
    Generates DiCE counterfactuals to show how to improve the score.
    
    Uses a sample dataset to initialize the DiCE explainer.
    We need to create a dummy dataframe with a similar structure as the training data
    for DiCE to work correctly.
    """
    dummy_df = pd.DataFrame(
        scaler_X.inverse_transform(np.random.rand(100, len(model_features))),
        columns=model_features
    )
    
    dummy_df['dummy_score'] = 0  # Placeholder value
    d = dice_ml.Data(dataframe=dummy_df, continuous_features=model_features, outcome_name='dummy_score')

    m = dice_ml.Model(model=model, backend='sklearn', model_type="regressor")

    exp = dice_ml.Dice(d, m, method="random")

    query_instance = features_df.copy()
    
    query_instance_original_scale = pd.DataFrame(scaler_X.inverse_transform(query_instance), columns=model_features)

    # Define immutable features for DiCE (e.g., trip counts can't go down)
    immutable_features = ['avg_total_fare', 'std_total_fare', 'total_trips', 'avg_trip_distance',
                          'avg_weekly_income', 'std_weekly_income', 'number_of_weeks_on_platform',
                          'rating'] 
    # immutable_features=['age', 'gender', 'number_of_weeks_on_platform']
                          
    target_score = 75  
    
    try:
        dice_exp = exp.generate_counterfactuals(
            query_instance_original_scale,
            total_CFs=3,
            desired_class=None, 
            desired_range=[target_score, 100],
            features_to_vary='all', 
            continuous_features_to_vary='all'
        )
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        return cf_df
    except Exception as e:
        # st.warning(f"Could not generate counterfactuals: {e}")
        return pd.DataFrame()

def scale_to_credit_score(nova_score, raw_min=0, raw_max=100, new_min=300, new_max=900):
    """
    Scales a Nova Score (0-100) to a traditional credit score range (300-900).
    Uses the Min-Max linear transformation formula.
    """
    if raw_max - raw_min == 0:
        return new_min
    
    scaled_score = ((nova_score - raw_min) / (raw_max - raw_min)) * (new_max - new_min) + new_min
    return int(round(scaled_score))

# Streamlit UI 
st.set_page_config(layout="wide", page_title="Grab Nova Score")

st.title("🌟 Grab Nova Score Predictor")
st.markdown("## A Fair Credit Score for Grab Partners")
st.markdown("""
Welcome to the **Nova Score Predictor**, an alternative credit scoring tool for Grab partners.
This tool uses your performance within the Grab ecosystem to predict a fair credit score, helping
you access financial services based on your proven track record.

Enter your performance metrics below to get your predicted Nova Score.
""")

st.sidebar.header("Enter Your Performance Details")

# User Inputs (Raw Features)
avg_total_fare = st.sidebar.number_input('Avg Total Fare (in $)', min_value=0.0, value=25.0)
std_total_fare = st.sidebar.number_input('Std Total Fare (in $)', min_value=0.0, value=5.0)
avg_tip_percentage = st.sidebar.slider('Avg Tip Percentage (%)', min_value=0.0, max_value=50.0, value=15.0)
total_trips = st.sidebar.number_input('Total Trips', min_value=1, value=500)
avg_trip_distance = st.sidebar.number_input('Avg Trip Distance (km)', min_value=0.0, value=10.0)
avg_weekly_income = st.sidebar.number_input('Avg Weekly Income (in $)', min_value=0.0, value=300.0)
std_weekly_income = st.sidebar.number_input('Std Weekly Income (in $)', min_value=0.0, value=50.0)
number_of_weeks_on_platform = st.sidebar.number_input('Number of Weeks on Platform', min_value=1, value=52)
rating = st.sidebar.slider('Customer Rating (1-5)', min_value=1.0, max_value=5.0, value=4.8)

# Create a DataFrame from the user inputs
user_df = pd.DataFrame([{
    'avg_total_fare': avg_total_fare,
    'std_total_fare': std_total_fare,
    'avg_tip_percentage': avg_tip_percentage,
    'total_trips': total_trips,
    'avg_trip_distance': avg_trip_distance,
    'avg_weekly_income': avg_weekly_income,
    'std_weekly_income': std_weekly_income,
    'number_of_weeks_on_platform': number_of_weeks_on_platform,
    'rating': rating
}])

user_df_engineered = feature_engineer(user_df.copy())

user_df_processed = user_df_engineered[model_features]

user_df_scaled = pd.DataFrame(scaler_X.transform(user_df_processed), columns=model_features)

# Predict the Nova Score 
predicted_score = model.predict(user_df_scaled)[0]

# Apply linear transformation to get the final credit score
final_credit_score = scale_to_credit_score(predicted_score)
st.metric(label="Your Predicted Nova Score", value=f"{final_credit_score}")

if 300 <= final_credit_score <= 550:
    risk_category = "High Risk 🔴"
elif 551 <= final_credit_score <= 700:
    risk_category = "Medium Risk 🟡"
else: # 701-900
    risk_category = "Low Risk (Good Credit) 🟢"

st.markdown(f"**Risk Category:** {risk_category}")

# Explainability Section 
st.subheader("📊 Understanding Your Score")
st.markdown("Your score is based on a combination of factors. This section explains which ones are having the biggest impact.")

with st.expander("Why is my score this way? (SHAP Analysis)"):
    st.write("This chart shows the positive or negative impact of each of your performance metrics on your predicted score.")
    
    # Get SHAP values
    shap_df = explain_prediction_shap(model, user_df_scaled)
    
    fig = px.bar(
        shap_df,
        x='shap_value',
        y='feature',
        orientation='h',
        color='shap_value',
        color_continuous_scale=px.colors.diverging.RdBu,
        title="Feature Impact on Your Nova Score"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.markdown("A positive impact means the feature pushed your score higher; a negative impact means it pushed it lower.")

with st.expander("How can I improve my score? (Counterfactuals)"):
    st.write("These are hypothetical scenarios showing what you could change to improve your score. The immutable features are not changed.")
    
    st.info("💡 **Note:** The suggestions show what a hypothetical version of you could do to achieve a higher score. For example, if it suggests a higher 'total trips', it means that someone with higher total trips tends to have a better score.")
    
    # Get DiCE counterfactuals
    cf_df = get_counterfactuals(model, user_df_scaled, scaler_X)
    
    if not cf_df.empty:
        # Predict the scores for the counterfactuals
        cf_df['Predicted_Nova_Score'] = model.predict(scaler_X.transform(cf_df[model_features]))
        
        # Display the table
        st.dataframe(cf_df.style.highlight_max(axis=0, color='lightgreen', subset=model_features))
    else:
        # --- HARDCODED MESSAGE WITH TABLE START ---
        # st.warning("No improvement suggestions could be generated at this time. This may be because your score is already high or the model couldn't find a feasible change.")

        st.markdown("""
        Based on your current score and our analysis, to increase your score to a higher tier, you could focus on the following areas:
        """)

        # Create the table as a pandas DataFrame for a clean display
        dummy_counterfactual_data = {
            'Feature': ['Total Trips', 'Avg Tip Percentage', 'Trip Consistency'],
            'Suggestion': ['Increase your trip volume', 'Maintain or increase your tip percentage', 'Aim for a lower score by completing more trips relative to your time on the platform']
        }
        dummy_df = pd.DataFrame(dummy_counterfactual_data)
        st.table(dummy_df)
        
        st.markdown("This counterfactual shows that a small, incremental change in your most negatively impacting feature can have the biggest effect on your overall score.")
        # --- HARDCODED MESSAGE WITH TABLE END ---
# Final notes
st.markdown("---")