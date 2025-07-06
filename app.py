import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set Streamlit page config
st.set_page_config(page_title="Aviation Analytics Dashboard", layout="wide")

# Load data (with cache)
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()

# Label encoding for classification (fit only on train to avoid leakage)
def label_encode_cols(df, encoders=None):
    le_cols = ['Class', 'Type of Travel', 'Gender']
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in le_cols if col in df.columns}
    for col in le_cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
    return df, encoders

# Sidebar filters
st.sidebar.title("Filters")

# Filter choices (before encoding)
classes = ['All'] + sorted(train['Class'].dropna().unique().tolist())
types = ['All'] + sorted(train['Type of Travel'].dropna().unique().tolist())
satisfactions = ['All'] + sorted(train['satisfaction'].dropna().unique().tolist())

selected_class = st.sidebar.selectbox("Select Class", classes)
selected_type = st.sidebar.selectbox("Select Type of Travel", types)
selected_satisfaction = st.sidebar.selectbox("Select Satisfaction", satisfactions)

# Filter data based on selections
filtered_data = train.copy()
if selected_class != 'All':
    filtered_data = filtered_data[filtered_data['Class'] == selected_class]
if selected_type != 'All':
    filtered_data = filtered_data[filtered_data['Type of Travel'] == selected_type]
if selected_satisfaction != 'All':
    filtered_data = filtered_data[filtered_data['satisfaction'] == selected_satisfaction]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Data Overview",
    "Visualizations",
    "Clustering",
    "Classification"
])

# --- Home Page ---
if page == "Home":
    st.title("Aviation Analytics Dashboard")
    st.write("""
    Welcome to the Airline Passenger Analytics Dashboard!
    - Use the sidebar to apply filters and explore the data.
    - Navigate to see overviews, visualizations, clustering, or try classification.
    - Data from `train.csv` and `test.csv`.
    """)

# --- Data Overview ---
elif page == "Data Overview":
    st.header("Filtered Data Overview")
    st.write(filtered_data.head(20))
    st.write("Shape:", filtered_data.shape)
    st.write("Columns:", list(filtered_data.columns))

# --- Visualizations ---
elif page == "Visualizations":
    st.title("Filtered Visualizations")

    # 1. Flight Distance Histogram
    st.subheader("Passenger Distribution by Flight Distance")
    plt.figure(figsize=(8,4))
    filtered_data['Flight Distance'].plot(kind='hist', bins=30)
    plt.xlabel('Flight Distance')
    plt.ylabel('Number of Passengers')
    plt.title('Distribution of Flight Distance')
    st.pyplot(plt.gcf())
    plt.clf()

    # 2. Class-wise Passenger Distribution
    st.subheader("Passenger Distribution by Class")
    class_counts = filtered_data['Class'].value_counts()
    plt.figure(figsize=(6,4))
    class_counts.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Passengers by Class (Filtered)')
    st.pyplot(plt.gcf())
    plt.clf()

    # 3. Type of Travel Distribution
    st.subheader("Business vs. Personal Travelers")
    travel_counts = filtered_data['Type of Travel'].value_counts()
    plt.figure(figsize=(6,6))
    travel_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Travel Purpose Distribution')
    st.pyplot(plt.gcf())
    plt.clf()

    # 4. Ancillary Service Ratings
    st.subheader("Ancillary Service Ratings")
    ancillary_services = [
        'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Baggage handling'
    ]
    for service in ancillary_services:
        if service in filtered_data.columns:
            st.write(f"**{service}**")
            plt.figure(figsize=(5,3))
            filtered_data[service].value_counts().sort_index().plot(kind='bar')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.title(f'{service} Rating')
            st.pyplot(plt.gcf())
            plt.clf()

    # 5. Satisfaction by Segment (Class)
    if 'satisfaction' in filtered_data.columns:
        st.subheader("Satisfaction by Class")
        plt.figure(figsize=(8,4))
        sns.countplot(x='Class', hue='satisfaction', data=filtered_data)
        plt.title('Satisfaction by Class')
        st.pyplot(plt.gcf())
        plt.clf()

    # 6. Correlation Heatmap
    st.subheader("Correlation Matrix (Numerical Features)")
    corr = filtered_data.corr(numeric_only=True)
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot(plt.gcf())
    plt.clf()

    # 7. Flight Delay Distribution
    if 'Departure Delay in Minutes' in filtered_data.columns:
        st.subheader("Departure Delay Distribution")
        plt.figure(figsize=(8,4))
        filtered_data['Departure Delay in Minutes'].plot(kind='hist', bins=30)
        plt.xlabel('Minutes')
        plt.title('Departure Delay Distribution')
        st.pyplot(plt.gcf())
        plt.clf()

# --- Clustering ---
elif page == "Clustering":
    st.title("K-Means Clustering: Segment Travelers by Behavior")
    st.write("We segment passengers based on Flight Distance and Age (where available).")

    cluster_data = filtered_data.dropna(subset=['Flight Distance', 'Age']).copy()
    if cluster_data.shape[0] > 0:
        features = ['Flight Distance', 'Age']
        X = cluster_data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans clustering (3 clusters as default)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
        cluster_data['Cluster'] = kmeans.labels_

        plt.figure(figsize=(8,6))
        colors = ['red', 'blue', 'green']
        for cluster in range(3):
            cluster_points = cluster_data[cluster_data['Cluster'] == cluster]
            plt.scatter(cluster_points['Flight Distance'], cluster_points['Age'],
                        label=f'Cluster {cluster}', alpha=0.6, color=colors[cluster])
        plt.xlabel('Flight Distance')
        plt.ylabel('Age')
        plt.title('Passenger Clusters by Flight Distance and Age')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("**Cluster Counts:**")
        st.write(cluster_data['Cluster'].value_counts())
    else:
        st.warning("Not enough data to perform clustering. Try adjusting the filters.")

# --- Classification ---
elif page == "Classification":
    st.title("Predict Passenger Satisfaction (Classification)")

    st.write("""
    **This module uses a Random Forest Classifier to predict if a passenger will be satisfied.**
    - Uses: Age, Gender, Class, Type of Travel, Flight Distance, Inflight wifi, Seat comfort, Food and drink, Baggage handling, Departure Delay, Arrival Delay
    """)

    # --- Prepare features and encode ---
    features = [
        'Age', 'Gender', 'Class', 'Type of Travel', 'Flight Distance',
        'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Baggage handling',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    data = train.dropna(subset=features + ['satisfaction']).copy()
    X_raw = data[features].copy()
    y = data['satisfaction']

    # Encode features
    X_encoded, encoders = label_encode_cols(X_raw)
    y_enc = LabelEncoder().fit_transform(y)  # satisfied/neutral or dissatisfied

    # --- Train/Test split and model training ---
    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_enc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    # --- Metrics ---
    acc = accuracy_score(y_val, y_pred)
    st.write(f"**Model Validation Accuracy:** {acc:.2%}")

    cm = confusion_matrix(y_val, y_pred)
    st.write("**Confusion Matrix:**")
    st.write(pd.DataFrame(cm))

    st.write("**Classification Report:**")
    st.text(classification_report(y_val, y_pred))

    # --- Prediction form ---
    st.write("---")
    st.header("Try a Prediction")
    # User input form
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

    # Encode user input
    user_input_df = pd.DataFrame([input_dict])
    user_input_df, _ = label_encode_cols(user_input_df, encoders)

    if st.button("Predict Satisfaction"):
        pred = clf.predict(user_input_df)[0]
        label = LabelEncoder().fit(y).inverse_transform([pred])[0]
        st.success(f"Prediction: The model predicts this passenger will be **{label.upper()}**.")

st.info("This dashboard features analytics, clustering (segmentation), and a live classifier. You can expand with new features, ML models, and uploads—just ask for help!")
