import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
MAINTENANCE_THRESHOLD = 0.25

def validate_data(df):
    """Validate uploaded CSV structure with comprehensive checks"""
    required_base_columns = {
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    }
    
    missing_columns = required_base_columns - set(df.columns)
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()
        
    # Check for at least one failure column if present
    failure_columns = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    has_failure_data = any(col in df.columns for col in failure_columns)
    
    return has_failure_data

@st.cache_data
def load_and_preprocess(uploaded_file):
    """Load and preprocess data with robust error handling"""
    try:
        df = pd.read_csv(uploaded_file)
        has_failure_data = validate_data(df)
        
        # Create engineered features safely
        try:
            df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
        except KeyError:
            st.error("Missing temperature columns for feature engineering")
            st.stop()
            
        try:
            df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / 9.5488
        except KeyError:
            st.error("Missing torque or rotation columns for power calculation")
            st.stop()
            
        try:
            df['stress_factor'] = df['Tool wear [min]'] * df['Rotational speed [rpm]']
        except KeyError:
            st.error("Missing tool wear or rotation columns for stress factor")
            st.stop()

        # Generate Product ID if missing
        if 'Product ID' not in df.columns:
            df['Product ID'] = [f"ID_{i+1:04d}" for i in range(len(df))]
            
        # Create failure target if available
        if has_failure_data:
            failure_cols = [col for col in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'] 
                          if col in df.columns]
            df['failure'] = df[failure_cols].any(axis=1).astype(int)
            
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

@st.cache_resource
def build_model(_df):
    """Build dynamic pipeline based on available columns"""
    # Base features that must exist after preprocessing
    guaranteed_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]',
        'Tool wear [min]', 'temp_diff', 'power', 'stress_factor'
    ]
    
    # Filter to only existing columns
    numeric_features = [col for col in guaranteed_features if col in _df.columns]
    categorical_features = ['Type'] if 'Type' in _df.columns else []

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'  # Ignore unexpected columns
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=8,
            random_state=42
        ))
    ])
    return model

def generate_maintenance_schedule(df, probabilities):
    """Generate maintenance recommendations"""
    schedule = pd.DataFrame({
        'Product ID': df['Product ID'],
        'Failure Probability': probabilities,
        'Recommended Maintenance': np.where(
            probabilities > MAINTENANCE_THRESHOLD,
            'Schedule within 48 hours',
            'Routine check recommended'
        ),
        'Criticality Level': pd.cut(
            probabilities,
            bins=[0, 0.2, 0.4, 0.6, 1],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
    })
    return schedule.sort_values('Failure Probability', ascending=False)

def show_shap_analysis(model, X_sample):
    """Display SHAP feature importance analysis"""
    try:
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        X_sample_transformed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()
        
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample_transformed)
        
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values
        
        if shap_values_class1.shape != X_sample_transformed.shape:
            st.error(f"Shape mismatch: SHAP values {shap_values_class1.shape} vs Data {X_sample_transformed.shape}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_class1,
                         X_sample_transformed,
                         feature_names=feature_names,
                         show=False)
        plt.title("Feature Impact on Failure Predictions", fontsize=14)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error generating SHAP plot: {str(e)}")

def main():
    st.set_page_config(
        page_title="Predictive Maintenance Analyzer",
        page_icon="🔧",
        layout="wide"
    )

    st.title("🔧 CSV-based Predictive Maintenance Analyzer")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload equipment data (CSV)", type="csv")
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started")
        st.stop()

    # Load and process data
    with st.spinner("Analyzing your data..."):
        df = load_and_preprocess(uploaded_file)
        model = build_model(df)

        # Handle target variable
        if 'failure' in df.columns:
            y = df['failure']
            has_real_data = True
        else:
            st.warning("No failure data found - using synthetic labels for demonstration")
            y = pd.Series(np.random.randint(0, 2, size=len(df)))
            has_real_data = False

        # Prepare features dynamically
        feature_cols = [
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'temp_diff', 'power', 'stress_factor'
        ]
        X = df[[col for col in feature_cols if col in df.columns]]
        
        # Train model with enhanced validation
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y if len(y.unique()) > 1 else None, random_state=42
            )
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X)[:, 1]
            maintenance_schedule = generate_maintenance_schedule(df, probabilities)
        except ValueError as e:
            st.error(f"Model training failed: {str(e)}")
            st.stop()

    # Navigation sidebar
    st.sidebar.title("Analysis Options")
    page = st.sidebar.radio("Select View", [
        "Data Overview",
        "Machine Health Check",
        "Maintenance Schedule",
        "Model Insights"
    ])

    if page == "Data Overview":
        st.header("📊 Data Summary")
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Basic Statistics**")
            st.write(df.describe())
        
        with col2:
            st.write("**Data Health Check**")
            st.write(pd.DataFrame({
                'Missing Values': df.isna().sum(),
                'Unique Values': df.nunique()
            }))

    elif page == "Machine Health Check":
        st.header("🔍 Machine Health Checker")
        product_ids = df['Product ID'].unique().tolist()
        product_id = st.selectbox("Select Equipment ID", product_ids)
        
        machine_data = df[df['Product ID'] == product_id]
        maintenance_info = maintenance_schedule[maintenance_schedule['Product ID'] == product_id]

        if not maintenance_info.empty:
            record = maintenance_info.iloc[0]
            machine_stats = machine_data.iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Maintenance Recommendation")
                st.metric("Failure Probability", f"{record['Failure Probability']:.1%}")
                st.metric("Criticality Level", record['Criticality Level'])
                st.write(f"**Action Required:** {record['Recommended Maintenance']}")

                prob = record['Failure Probability']
                gauge = '🟥' * int(prob * 10) + '⬜' * (10 - int(prob * 10))
                st.write(f"**Risk Indicator:** {gauge} {prob:.0%}")

            with col2:
                st.subheader("Equipment Parameters")
                params = {
                    'Type': machine_stats.get('Type', 'N/A'),
                    'Tool Wear': f"{machine_stats.get('Tool wear [min]', 'N/A')} mins",
                    'Rotational Speed': f"{machine_stats.get('Rotational speed [rpm]', 'N/A')} RPM",
                    'Temperature Diff': f"{machine_stats.get('temp_diff', 'N/A'):.1f}°K"
                }
                for k, v in params.items():
                    st.write(f"**{k}:** {v}")

    elif page == "Maintenance Schedule":
        st.header("📅 Maintenance Priorities")
        critical_filter = st.multiselect(
            "Filter by Criticality Level",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High']
        )
        
        filtered = maintenance_schedule[maintenance_schedule['Criticality Level'].isin(critical_filter)]
        st.dataframe(filtered, use_container_width=True)
        
        st.download_button(
            "Export Schedule",
            data=filtered.to_csv(index=False).encode('utf-8'),
            file_name='maintenance_schedule.csv',
            mime='text/csv'
        )

    elif page == "Model Insights":
        st.header("📈 Model Performance Analysis")
        
        if has_real_data:
            with st.expander("Classification Report"):
                y_pred = model.predict(X_test)
                st.write(classification_report(y_test, y_pred, output_dict=True))
            
            with st.expander("Confusion Matrix"):
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True,
                               labels=dict(x="Predicted", y="Actual"),
                               title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Feature Importance"):
            try:
                X_sample = X.sample(100, random_state=42)
                show_shap_analysis(model, X_sample)
            except Exception as e:
                st.error(f"Couldn't generate feature importance: {str(e)}")

if __name__ == '__main__':
    main()