import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt

# Configuration
MAINTENANCE_THRESHOLD = 0.25

def load_and_preprocess(file_path):
    """Load and preprocess the data"""
    df = pd.read_csv(file_path)
    failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df['failure'] = df[failure_cols].any(axis=1).astype(int)
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / 9.5488
    df['stress_factor'] = df['Tool wear [min]'] * df['Rotational speed [rpm]']
    return df

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
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
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

def display_welcome():
    """Display interactive welcome message"""
    print("""
    ███╗   ███╗ █████╗ ██╗███╗   ██╗████████╗███████╗███╗   ██╗ █████╗ ███╗   ██╗ ██████╗ ███████╗
    ████╗ ████║██╔══██╗██║████╗  ██║╚══██╔══╝██╔════╝████╗  ██║██╔══██╗████╗  ██║██╔════╝ ██╔════╝
    ██╔████╔██║███████║██║██╔██╗ ██║   ██║   █████╗  ██╔██╗ ██║███████║██╔██╗ ██║██║  ███╗█████╗
    ██║╚██╔╝██║██╔══██║██║██║╚██╗██║   ██║   ██╔══╝  ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██╔══╝
    ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║   ██║   ███████╗██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝███████╗
    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝
    """)
    print("Welcome to Smart Maintenance Predictor!")
    print("🔧 Predict machine failures | 📅 Schedule maintenance | 📊 Analyze equipment health\n")
    input("Press Enter to continue...")

def display_data_insights(df):
    """Show interactive data exploration"""
    print("\n📊 Dataset Insights:")
    print(f"Total Machines: {len(df)}")
    print(f"Total Failures: {df['failure'].sum()} ({df['failure'].mean():.1%})")

    failure_types = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()
    print("\n🔧 Failure Type Distribution:")
    print(failure_types.to_string())

    plt.figure(figsize=(10, 5))
    failure_types.plot(kind='bar', color='#ff6666')
    plt.title("Failure Type Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def interactive_maintenance_check(maintenance_schedule, df):
    """Enhanced interactive machine check"""
    while True:
        print("\n" + "="*50)
        product_id = input("\nEnter Product ID (e.g. L47181) or 'back': ").strip().upper()

        if product_id.lower() == 'back':
            break

        machine_data = df[df['Product ID'] == product_id]
        maintenance_info = maintenance_schedule[maintenance_schedule['Product ID'] == product_id]

        if not maintenance_info.empty:
            record = maintenance_info.iloc[0]
            machine_stats = machine_data.iloc[0]

            print(f"\n🔍 Maintenance Report for {product_id}")
            print(f"📈 Failure Probability: {record['Failure Probability']:.1%}")
            print(f"🛠️  Recommended Action: {record['Recommended Maintenance']}")
            print(f"🚨 Criticality Level: {record['Criticality Level']}")

            print("\n⚙️ Machine Parameters:")
            print(f"Type: {machine_stats['Type']}")
            print(f"Tool Wear: {machine_stats['Tool wear [min]']} mins")
            print(f"Rotational Speed: {machine_stats['Rotational speed [rpm]']} RPM")
            print(f"Temperature Difference: {machine_stats['temp_diff']:.1f}°K")

            prob = record['Failure Probability']
            gauge = '🟥' * int(prob*10) + '⬜' * (10 - int(prob*10))
            print(f"\n📊 Risk Gauge: {gauge} {prob:.0%}")

            if prob > 0.6:
                print("\n🚨 IMMEDIATE ACTION REQUIRED! Failure risk critical!")
            elif prob > 0.4:
                print("\n⚠️ High Priority: Schedule maintenance within 48 hours")
            elif prob > 0.2:
                print("\nℹ️ Medium Priority: Schedule within next week")
            else:
                print("\n✅ Normal Operation: Routine monitoring sufficient")

            similar = maintenance_schedule[
                (maintenance_schedule['Criticality Level'] == record['Criticality Level']) &
                (maintenance_schedule['Product ID'] != product_id)
            ].head(3)

            if not similar.empty:
                print("\n🔗 Similar Machines Needing Attention:")
                print(similar[['Product ID', 'Failure Probability']].to_string(index=False))

        else:
            print(f"\n⚠️ No records found for: {product_id}")
            close_matches = df[df['Product ID'].str.startswith(product_id[:2])]['Product ID'].unique()[:5]
            print("Try these similar IDs:", ", ".join(close_matches))

def display_maintenance_priorities(schedule):
    """Interactive priority display"""
    print("\n📅 Maintenance Priorities:")
    print("1. Show Critical Issues")
    print("2. Show High Priority")
    print("3. Show Medium Priority")
    print("4. Show All Priorities")
    print("5. Export to CSV")

    choice = input("\nSelect display option (1-5): ").strip()

    # Validate input first
    if choice not in {'1', '2', '3', '4', '5'}:
        print("Invalid choice!")
        return

    if choice == '5':
        schedule.to_csv('maintenance_schedule.csv', index=False)
        print("✅ Schedule exported to maintenance_schedule.csv")
        return

    # Get filters after validation
    filters = {
        '1': 'Critical',
        '2': 'High',
        '3': 'Medium',
        '4': None
    }[choice]

    filtered = schedule if filters is None else schedule[schedule['Criticality Level'] == filters]

    n = input("How many entries to display? (Enter for all): ").strip()
    n = int(n) if n.isdigit() else len(filtered)

    print(f"\n🔝 Top {n} Maintenance Priorities:")
    print(filtered.head(n).to_string(index=False))

    plt.figure(figsize=(10, 4))
    filtered['Criticality Level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Criticality Distribution")
    plt.show()

def main_menu():
    """Interactive main menu system"""
    df = load_and_preprocess('ai4i2020.csv')
    model = build_model()

    failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=failure_cols + ['failure', 'UDI', 'Product ID'])
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X)[:, 1]
    maintenance_schedule = generate_maintenance_schedule(df, probabilities)

    while True:
        print("\033[0;32m\n" + "="*50 + "\033[0m")  # Green separator
        print("\033[0;32m🏠 Main Menu\033[0m")
        print("\033[0;32m1. 📊 View Dataset Insights\033[0m")
        print("\033[0;32m2. 🔍 Check Machine Status\033[0m")
        print("\033[0;32m3. 📅 View Maintenance Schedule\033[0m")
        print("\033[0;32m4. 📈 View Model Performance\033[0m")
        print("\033[0;32m5. 🚪 Exit\033[0m")

        choice = input("\033[0;32m\nSelect an option (1-5): \033[0m").strip()

        if choice == '1':
            display_data_insights(df)
        elif choice == '2':
            interactive_maintenance_check(maintenance_schedule, df)
        elif choice == '3':
            display_maintenance_priorities(maintenance_schedule)
        elif choice == '4':
            y_pred = model.predict(X_test)
            print("\n📈 Model Performance Report:")
            print(classification_report(y_test, y_pred))
            print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

            # Get proper feature names
            preprocessor = model.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()

            # Use a sample for SHAP calculations
            sample_idx = np.random.choice(X_test.index, 100, replace=False)
            X_sample = X.loc[sample_idx]

            # Transform data using fitted preprocessor
            X_processed = preprocessor.transform(X_sample)

            # Get SHAP values
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            shap_values = explainer.shap_values(X_processed)

            # Plot with correct feature names
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values,
                           X_processed,
                             feature_names=feature_names,
                              plot_type='bar',
                              show=False)
            plt.title('Feature Importance for Maintenance Predictions')
            plt.tight_layout()
            plt.show()
        elif choice == '5':
            print("\n👋 Thank you for using Smart Maintenance Predictor!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == '__main__':
    display_welcome()
    main_menu()