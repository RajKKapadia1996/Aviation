import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set Streamlit page config
st.set_page_config(page_title="Aviation Analytics Dashboard", layout="wide")

# Load data (with cache)
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()

# Sidebar filters
st.sidebar.title("Filters")

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
    "Clustering"
])

# --- Home Page ---
if page == "Home":
    st.title("Aviation Analytics Dashboard")
    st.write("""
    Welcome to the Airline Passenger Analytics Dashboard!
    - Use the sidebar to apply filters and explore the data.
    - Navigate to see overviews, visualizations, or clustering results.
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
    plt.title('Passengers by Class (Filtered)")
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

st.info("You can expand this dashboard with more features, such as custom clustering, predictions, or other interactive analytics. If you need help, just ask!")
