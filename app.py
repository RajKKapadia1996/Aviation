import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")  # Global style

# Set Streamlit page config
st.set_page_config(page_title="Aviation Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()

def label_encode_cols(df, encoders=None):
    le_cols = ['Class', 'Type of Travel', 'Gender']
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in le_cols if col in df.columns}
    for col in le_cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
    return df, encoders

st.sidebar.title("Filters")
classes = ['All'] + sorted(train['Class'].dropna().unique().tolist())
types = ['All'] + sorted(train['Type of Travel'].dropna().unique().tolist())
satisfactions = ['All'] + sorted(train['satisfaction'].dropna().unique().tolist())

selected_class = st.sidebar.selectbox("Select Class", classes)
selected_type = st.sidebar.selectbox("Select Type of Travel", types)
selected_satisfaction = st.sidebar.selectbox("Select Satisfaction", satisfactions)

filtered_data = train.copy()
if selected_class != 'All':
    filtered_data = filtered_data[filtered_data['Class'] == selected_class]
if selected_type != 'All':
    filtered_data = filtered_data[filtered_data['Type of Travel'] == selected_type]
if selected_satisfaction != 'All':
    filtered_data = filtered_data[filtered_data['satisfaction'] == selected_satisfaction]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Data Overview",
    "Visualizations",
    "Clustering",
    "Classification",
    "Regression"
])

# Home
if page == "Home":
    st.title("Aviation Analytics Dashboard ✈️")
    st.markdown("""
    Welcome to the **Aviation Analytics Dashboard**!
    - 🌎 Use the sidebar to filter and explore the data.
    - 📊 View beautiful visualizations, clustering, classification, and regression.
    - 🏆 All features are interactive and ready for your exploration!
    """)

# Data Overview
elif page == "Data Overview":
    st.header("Filtered Data Overview")
    st.dataframe(filtered_data.head(20), use_container_width=True)
    st.markdown(f"**Shape:** {filtered_data.shape}")
    st.markdown(f"**Columns:** {', '.join(filtered_data.columns)}")

# Visualizations
elif page == "Visualizations":
    st.title("🌈 Beautiful Data Visualizations")

    # 1. Passenger Distribution by Flight Distance
    st.subheader("1️⃣ Passenger Distribution by Flight Distance")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(filtered_data['Flight Distance'], bins=30, kde=True, color="#0174DF", ax=ax)
    ax.set_xlabel('Flight Distance')
    ax.set_ylabel('Number of Passengers')
    ax.set_title('Distribution of Flight Distance', fontsize=16, fontweight='bold')
    st.pyplot(fig)

    # 2. Passenger Distribution by Class
    st.subheader("2️⃣ Passenger Distribution by Class")
    fig, ax = plt.subplots(figsize=(7,4))
    palette = sns.color_palette("crest")
    sns.countplot(x='Class', data=filtered_data, order=filtered_data['Class'].value_counts().index, palette=palette, ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Passengers by Class', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # 3. Business vs. Personal Travelers
    st.subheader("3️⃣ Business vs. Personal Travelers")
    fig, ax = plt.subplots(figsize=(6,6))
    travel_counts = filtered_data['Type of Travel'].value_counts()
    colors = sns.color_palette('pastel')
    ax.pie(travel_counts, labels=travel_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 13})
    ax.set_title('Travel Purpose Distribution', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # 4. Ancillary Service Ratings
    st.subheader("4️⃣ Ancillary Service Ratings")
    ancillary_services = [
        'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Baggage handling'
    ]
    for service in ancillary_services:
        if service in filtered_data.columns:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.countplot(x=service, data=filtered_data, palette="ch:s=.25,rot=-.25", ax=ax)
            ax.set_xlabel(f'{service} Rating')
            ax.set_ylabel('Count')
            ax.set_title(f'{service} Rating Distribution', fontsize=12, fontweight='bold')
            st.pyplot(fig)

    # 5. Satisfaction by Class
    if 'satisfaction' in filtered_data.columns:
        st.subheader("5️⃣ Satisfaction by Class")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x='Class', hue='satisfaction', data=filtered_data, palette='flare', ax=ax)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Satisfaction by Class', fontsize=14, fontweight='bold')
        st.pyplot(fig)

    # 6. Correlation Matrix
    st.subheader("6️⃣ Correlation Matrix (Numerical Features)")
    corr = filtered_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11,7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # 7. Departure Delay Distribution
    if 'Departure Delay in Minutes' in filtered_data.columns:
        st.subheader("7️⃣ Departure Delay Distribution")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(filtered_data['Departure Delay in Minutes'], bins=30, kde=True, color="#FA5858", ax=ax)
        ax.set_xlabel('Departure Delay (Minutes)')
        ax.set_ylabel('Number of Flights')
        ax.set_title('Departure Delay Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)

# Clustering
elif page == "Clustering":
    st.title("🎯 K-Means Clustering: Segment Travelers by Behavior")
    st.write("Passengers segmented by **Flight Distance** and **Age** (where available).")

    cluster_data = filtered_data.dropna(subset=['Flight Distance', 'Age']).copy()
    if cluster_data.shape[0] > 0:
        features = ['Flight Distance', 'Age']
        X = cluster_data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        cluster_data['Cluster'] = kmeans.labels_

        fig, ax = plt.subplots(figsize=(8,6))
        palette = sns.color_palette("husl", 3)
        for cluster in range(3):
            cluster_points = cluster_data[cluster_data['Cluster'] == cluster]
            ax.scatter(cluster_points['Flight Distance'], cluster_points['Age'],
                       label=f'Cluster {cluster}', alpha=0.6, s=60, color=palette[cluster])
        ax.set_xlabel('Flight Distance')
        ax.set_ylabel('Age')
        ax.set_title('Passenger Clusters by Flight Distance and Age', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)

        st.markdown("**Cluster Counts:**")
        st.dataframe(cluster_data['Cluster'].value_counts().rename('Count'))

    else:
        st.warning("Not enough data to perform clustering. Try adjusting the filters.")

# Classification
elif page == "Classification":
    st.title("🔍 Predict Passenger Satisfaction (Classification)")

    st.write("""
    **Random Forest Classifier predicts if a passenger will be satisfied.**
    - Features: Age, Gender, Class, Type of Travel, Flight Distance, Service Ratings, Delays
    """)

    features = [
        'Age', 'Gender', 'Class', 'Type of Travel', 'Flight Distance',
        'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Baggage handling',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    data = train.dropna(subset=features + ['satisfaction']).copy()
    X_raw = data[features].copy()
    y = data['satisfaction']

    X_encoded, encoders = label_encode_cols(X_raw)
    y_enc = LabelEncoder().fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_enc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    st.success(f"**Model Validation Accuracy:** {acc:.2%}")

    # Confusion matrix plot
    st.markdown("**Confusion Matrix:**")
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="flare", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # Feature importance plot
    st.markdown("**Feature Importance:**")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns
    fig, ax = plt.subplots(figsize=(9,5))
    sns.barplot(x=importances[indices], y=feature_names[indices], palette="crest", ax=ax)
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    st.write("**Classification Report:**")
    st.text(classification_report(y_val, y_pred))

    st.write("---")
    st.header("Try a Prediction")
    input_dict = {}
    input_dict['Age'] = st.number_input("Age", min_value=1, max_value=100, value=30)
    input_dict['Gender'] = st.selectbox("Gender", encoders['Gender'].classes_)
    input_dict['Class'] = st.selectbox("Class", encoders['Class'].classes_)
    input_dict['Type of Travel'] = st.selectbox("Type of Travel", encoders['Type of Travel'].classes_)
    input_dict['Flight Distance'] = st.number_input("Flight Distance", min_value=31, max_value=5000, value=500)
    input_dict['Inflight wifi service'] = st.slider("Inflight wifi service (0=bad, 5=excellent)", 0, 5, 3)
    input_dict['Seat comfort'] = st.slider("Seat comfort (0=bad, 5=excellent)", 0, 5, 3)
    input_dict['Food and drink'] = st.slider("Food and drink (0=bad, 5=excellent)", 0, 5, 3)
    input_dict['Baggage handling'] = st.slider("Baggage handling (0=bad, 5=excellent)", 0, 5, 3)
    input_dict['Departure Delay in Minutes'] = st.number_input("Departure Delay (minutes)", 0, 1000, 0)
    input_dict['Arrival Delay in Minutes'] = st.number_input("Arrival Delay (minutes)", 0, 1000, 0)

    user_input_df = pd.DataFrame([input_dict])
    user_input_df, _ = label_encode_cols(user_input_df, encoders)

    if st.button("Predict Satisfaction"):
        pred = clf.predict(user_input_df)[0]
        label = LabelEncoder().fit(y).inverse_transform([pred])[0]
        st.success(f"Prediction: The model predicts this passenger will be **{label.upper()}**.")

# Regression
elif page == "Regression":
    st.title("📈 Forecast Passenger Demand (Regression)")

    st.write("""
    **Linear Regression forecasts customer demand (proxy: Flight Distance) using passenger and flight features.**
    - Features: Age, Gender, Class, Type of Travel, Service Ratings, Delays
    """)

    reg_features = [
        'Age', 'Gender', 'Class', 'Type of Travel',
        'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Baggage handling',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    target = 'Flight Distance'
    reg_data = train.dropna(subset=reg_features + [target]).copy()
    X_reg_raw = reg_data[reg_features]
    y_reg = reg_data[target]

    X_reg_encoded, reg_encoders = label_encode_cols(X_reg_raw)
    X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(X_reg_encoded, y_reg, test_size=0.2, random_state=42)
    reg_model = LinearRegression()
    reg_model.fit(X_reg_train, y_reg_train)
    y_reg_pred = reg_model.predict(X_reg_val)

    rmse = np.sqrt(mean_squared_error(y_reg_val, y_reg_pred))
    r2 = r2_score(y_reg_val, y_reg_pred)
    st.success(f"**Validation RMSE:** {rmse:.2f} | **Validation R²:** {r2:.3f}")

    # Scatter plot: Actual vs. Predicted
    st.markdown("**Actual vs. Predicted Flight Distance:**")
    fig, ax = plt.subplots(figsize=(7,6))
    sns.scatterplot(x=y_reg_val, y=y_reg_pred, color='#5F04B4', s=70, ax=ax)
    ax.plot([y_reg_val.min(), y_reg_val.max()], [y_reg_val.min(), y_reg_val.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Flight Distance')
    ax.set_ylabel('Predicted Flight Distance')
    ax.set_title('Actual vs. Predicted', fontsize=13, fontweight='bold')
    st.pyplot(fig)

    # Residuals plot
    st.markdown("**Residuals Plot:**")
    residuals = y_reg_val - y_reg_pred
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(residuals, bins=30, color="#58ACFA", kde=True, ax=ax)
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_title('Residuals of Prediction', fontsize=13, fontweight='bold')
    st.pyplot(fig)

    st.write("---")
    st.header("Try a Demand Forecast (Flight Distance Prediction)")
    reg_input_dict = {}
    reg_input_dict['Age'] = st.number_input("Age", min_value=1, max_value=100, value=30, key="reg_age")
    reg_input_dict['Gender'] = st.selectbox("Gender", reg_encoders['Gender'].classes_, key="reg_gender")
    reg_input_dict['Class'] = st.selectbox("Class", reg_encoders['Class'].classes_, key="reg_class")
    reg_input_dict['Type of Travel'] = st.selectbox("Type of Travel", reg_encoders['Type of Travel'].classes_, key="reg_type")
    reg_input_dict['Inflight wifi service'] = st.slider("Inflight wifi service (0=bad, 5=excellent)", 0, 5, 3, key="reg_wifi")
    reg_input_dict['Seat comfort'] = st.slider("Seat comfort (0=bad, 5=excellent)", 0, 5, 3, key="reg_seat")
    reg_input_dict['Food and drink'] = st.slider("Food and drink (0=bad, 5=excellent)", 0, 5, 3, key="reg_food")
    reg_input_dict['Baggage handling'] = st.slider("Baggage handling (0=bad, 5=excellent)", 0, 5, 3, key="reg_baggage")
    reg_input_dict['Departure Delay in Minutes'] = st.number_input("Departure Delay (minutes)", 0, 1000, 0, key="reg_dep_delay")
    reg_input_dict['Arrival Delay in Minutes'] = st.number_input("Arrival Delay (minutes)", 0, 1000, 0, key="reg_arr_delay")

    reg_user_input_df = pd.DataFrame([reg_input_dict])
    reg_user_input_df, _ = label_encode_cols(reg_user_input_df, reg_encoders)

    if st.button("Forecast Flight Distance (as demand proxy)"):
        reg_pred = reg_model.predict(reg_user_input_df)[0]
        st.success(f"Forecasted Flight Distance (demand proxy): **{reg_pred:.2f} units**")

st.info("This dashboard now features beautiful analytics, interactive clustering, full-featured classification and regression—with professional visuals throughout! 🚀")
