import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Aviation Analytics Dashboard", layout="wide")

# Load the datasets
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()

# Sidebar navigation
st.sidebar.title("Aviation Data Dashboard")
page = st.sidebar.radio("Go to", [
    "Home",
    "Train Data Overview",
    "Test Data Overview",
    "Visualizations"
])

if page == "Home":
    st.title("Aviation Analytics Dashboard")
    st.write("""
    Welcome to the Airline Passenger Analytics Dashboard!
    - Explore passenger trends, satisfaction, service ratings, delays, and more.
    - Data from train.csv and test.csv.
    - Use the sidebar to navigate through views and visualizations.
    """)

elif page == "Train Data Overview":
    st.header("Training Data Overview")
    st.write(train.head())
    st.write("Shape:", train.shape)
    st.write("Columns:", list(train.columns))

elif page == "Test Data Overview":
    st.header("Test Data Overview")
    st.write(test.head())
    st.write("Shape:", test.shape)
    st.write("Columns:", list(test.columns))

elif page == "Visualizations":
    st.title("Visual Explorations")

    # 1. Passenger Demand Over Flight Distance
    st.subheader("1. Passenger Distribution by Flight Distance")
    plt.figure(figsize=(8,4))
    train['Flight Distance'].plot(kind='hist', bins=30)
    plt.xlabel('Flight Distance')
    plt.ylabel('Number of Passengers')
    plt.title('Distribution of Flight Distance')
    st.pyplot(plt.gcf())
    plt.clf()

    # 2. Class-wise Passenger Distribution
    st.subheader("2. Passenger Distribution by Class")
    class_counts = train['Class'].value_counts()
    plt.figure(figsize=(6,4))
    class_counts.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Passengers by Class')
    st.pyplot(plt.gcf())
    plt.clf()

    # 3. Type of Travel Distribution
    st.subheader("3. Business vs. Personal Travelers")
    travel_counts = train['Type of Travel'].value_counts()
    plt.figure(figsize=(6,6))
    travel_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Travel Purpose Distribution')
    st.pyplot(plt.gcf())
    plt.clf()

    # 4. Booking Lead Time (if present)
    if 'Lead Time' in train.columns:
        st.subheader("4. Booking Lead Time Distribution")
        plt.figure(figsize=(8,4))
        train['Lead Time'].plot(kind='hist', bins=20)
        plt.xlabel('Days Before Flight')
        plt.title('Booking Lead Time')
        st.pyplot(plt.gcf())
        plt.clf()

    # 5. Ancillary Service Ratings
    st.subheader("5. Ancillary Service Ratings")
    ancillary_services = [
        'Inflight wifi service', 'Seat comfort', 'Food and drink', 'Baggage handling'
    ]
    for service in ancillary_services:
        if service in train.columns:
            st.write(f"**{service}**")
            plt.figure(figsize=(5,3))
            train[service].value_counts().sort_index().plot(kind='bar')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.title(f'{service} Rating')
            st.pyplot(plt.gcf())
            plt.clf()

    # 6. Satisfaction by Segment (Class)
    if 'satisfaction' in train.columns:
        st.subheader("6. Satisfaction by Class")
        plt.figure(figsize=(8,4))
        sns.countplot(x='Class', hue='satisfaction', data=train)
        plt.title('Satisfaction by Class')
        st.pyplot(plt.gcf())
        plt.clf()

    # 7. Correlation Heatmap
    st.subheader("7. Correlation Matrix (Numerical Features)")
    corr = train.corr(numeric_only=True)
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot(plt.gcf())
    plt.clf()

    # 8. Flight Delay Distribution
    if 'Departure Delay in Minutes' in train.columns:
        st.subheader("8. Departure Delay Distribution")
        plt.figure(figsize=(8,4))
        train['Departure Delay in Minutes'].plot(kind='hist', bins=30)
        plt.xlabel('Minutes')
        plt.title('Departure Delay Distribution')
        st.pyplot(plt.gcf())
        plt.clf()

    # 9. Frequent Flyer Segment Analysis (by satisfaction)
    if 'satisfaction' in train.columns:
        st.subheader("9. Average Flight Distance by Satisfaction")
        avg_dist = train.groupby('satisfaction')['Flight Distance'].mean()
        plt.figure(figsize=(6,4))
        avg_dist.plot(kind='bar')
        plt.xlabel('Satisfaction')
        plt.ylabel('Avg Flight Distance')
        plt.title('Avg Flight Distance by Satisfaction')
        st.pyplot(plt.gcf())
        plt.clf()

st.info("You can expand this dashboard with interactive filters, predictive models, and clustering as next steps!")
