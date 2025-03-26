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

@st.cache_data
def load_and_preprocess(file_path='ai4i2020.csv'):
    """Load and preprocess the data"""
    df = pd.read_csv(file_path)
    failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df['failure'] = df[failure_cols].any(axis=1).astype(int)
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / 9.5488
    df['stress_factor'] = df['Tool wear [min]'] * df['Rotational speed [rpm]']
    return df

@st.cache_resource
def build_model():
    """Build the predictive maintenance pipeline"""
    numeric_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]',
        'Tool wear [min]', 'temp_diff', 'power', 'stress_factor'
    ]
    categorical_features = ['Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

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
        # Extract components from the pipeline
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform features using the preprocessor
        X_sample_transformed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()
        
        # Use TreeExplainer on the classifier
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample_transformed)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification case
            shap_values_class1 = shap_values[1]
        elif len(shap_values.shape) == 3:
            # Multi-class format array
            shap_values_class1 = shap_values[:, :, 1]
        else:
            # Single array format
            shap_values_class1 = shap_values
        
        # Validate shapes
        if shap_values_class1.shape != X_sample_transformed.shape:
            st.error(f"Shape mismatch: SHAP values {shap_values_class1.shape} vs Data {X_sample_transformed.shape}")
            return
        
        # Create summary plot
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
    # Page configuration
    st.set_page_config(
        page_title="Smart Maintenance Predictor",
        page_icon="🔧",
        layout="wide"
    )

    # App title and description
    st.title("🔧 Predictive Maintenance App")
    st.markdown("A tool for predicting machine failures, scheduling maintenance, and analyzing equipment health.")

    # Load data and model
    df = load_and_preprocess()
    model = build_model()

    # Train model
    failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=failure_cols + ['failure', 'UDI', 'Product ID'])
    y = df['failure']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X)[:, 1]
    maintenance_schedule = generate_maintenance_schedule(df, probabilities)

    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard",
        "Machine Check",
        "Maintenance Schedule",
        "Model Analysis"
    ])

    # Dashboard Page
    if page == "Dashboard":
        st.header("📊 Dataset Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Machines", len(df))
            st.metric("Total Failures", f"{df['failure'].sum()} ({df['failure'].mean():.1%})")
        
        with col2:
            failure_types = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()
            st.bar_chart(failure_types)

        # Interactive histogram
        df['Failure Status'] = df['failure'].map({0: 'No Failure', 1: 'Failure'})
        fig = px.histogram(df, x='Rotational speed [rpm]', color='Failure Status',
                          marginal="box", title="Rotational Speed Distribution by Failure Status")
        st.plotly_chart(fig, use_container_width=True)

    # Machine Check Page
    elif page == "Machine Check":
        st.header("🔍 Machine Status Check")
        
        product_ids = df['Product ID'].unique().tolist()
        product_id = st.selectbox("Select Product ID", product_ids)
        
        machine_data = df[df['Product ID'] == product_id]
        maintenance_info = maintenance_schedule[maintenance_schedule['Product ID'] == product_id]

        if not maintenance_info.empty:
            record = maintenance_info.iloc[0]
            machine_stats = machine_data.iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Maintenance Report")
                st.metric("Failure Probability", f"{record['Failure Probability']:.1%}")
                st.metric("Criticality Level", record['Criticality Level'])
                st.write(f"**Recommended Action:** {record['Recommended Maintenance']}")

                # Visual risk gauge
                prob = record['Failure Probability']
                gauge = '🟥' * int(prob * 10) + '⬜' * (10 - int(prob * 10))
                st.write(f"**Risk Gauge:** {gauge} {prob:.0%}")

            with col2:
                st.subheader("Machine Parameters")
                st.write(f"**Type:** {machine_stats['Type']}")
                st.write(f"**Tool Wear:** {machine_stats['Tool wear [min]']} mins")
                st.write(f"**Rotational Speed:** {machine_stats['Rotational speed [rpm]']} RPM")
                st.write(f"**Temperature Difference:** {machine_stats['temp_diff']:.1f}°K")

            # Similar machines section
            similar = maintenance_schedule[
                (maintenance_schedule['Criticality Level'] == record['Criticality Level']) &
                (maintenance_schedule['Product ID'] != product_id)
            ].head(3)
            if not similar.empty:
                st.subheader("Similar Machines Needing Attention")
                st.dataframe(similar[['Product ID', 'Failure Probability']], use_container_width=True)
        else:
            st.warning(f"No records found for Product ID: {product_id}")

    # Maintenance Schedule Page
    elif page == "Maintenance Schedule":
        st.header("📅 Maintenance Priorities")
        
        criticality_levels = maintenance_schedule['Criticality Level'].unique().tolist()
        selected_levels = st.multiselect("Filter by Criticality Level", criticality_levels, default=criticality_levels)
        
        filtered = maintenance_schedule[maintenance_schedule['Criticality Level'].isin(selected_levels)]
        
        if filtered.empty:
            st.info("No machines match the selected criticality levels.")
        else:
            st.dataframe(filtered, use_container_width=True)

        st.download_button(
            label="Export to CSV",
            data=filtered.to_csv(index=False).encode('utf-8'),
            file_name='maintenance_schedule.csv',
            mime='text/csv'
        )

    # Model Analysis Page
    elif page == "Model Analysis":
        st.header("📈 Model Performance Analysis")
        
        y_pred = model.predict(X_test)
        
        with st.expander("Classification Report", expanded=True):
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            st.markdown("""
            - **Precision:** Proportion of true positives among predicted positives
            - **Recall:** Proportion of true positives among actual positives
            - **F1-Score:** Harmonic mean of precision and recall
            """)

        with st.expander("Feature Importance Analysis", expanded=True):
            with st.spinner("Generating SHAP explanations..."):
                try:
                    X_sample = X_test.sample(100, random_state=42)
                    show_shap_analysis(model, X_sample)
                except Exception as e:
                    st.error(f"Feature importance analysis failed: {str(e)}")

        with st.expander("Confusion Matrix", expanded=True):
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True,
                          labels=dict(x="Predicted", y="Actual"),
                          x=['No Failure', 'Failure'],
                          y=['No Failure', 'Failure'],
                          title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            - **True Positives (TP):** Correct failure predictions
            - **True Negatives (TN):** Correct non-failure predictions
            - **False Positives (FP):** Incorrect failure predictions
            - **False Negatives (FN):** Missed failures
            """)

if __name__ == '__main__':
    main()