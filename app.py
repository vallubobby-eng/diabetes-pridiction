import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Streamlit Page Config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0d6efd;
    }
    .metric-label {
        color: #6c757d;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to ensure data exists
@st.cache_data
def ensure_data():
    if not os.path.exists('diabetes.csv'):
        try:
            url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
            columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            df = pd.read_csv(url, header=None, names=columns)
            df.to_csv('diabetes.csv', index=False)
            return True
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            return False
    return True

# Load and train model
@st.cache_resource
def load_and_train_model():
    ensure_data()
    try:
        df = pd.read_csv('diabetes.csv')
    except Exception as e:
        return None, None, 0, None

    X = df.drop('Outcome', axis=1)
    Y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=48)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    
    accuracy = model.score(X_test, Y_test)
    
    return model, scaler, accuracy, df

# Train the model
with st.spinner('Loading data and training model...'):
    model, scaler, accuracy, df = load_and_train_model()

# Sidebar Navigation
st.sidebar.title("🏥 HealthGuard")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Navigation", ["🏠 Home", "🔬 Diabetes Prediction", "📊 Data Exploration", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.info("This application uses Machine Learning to predict the likelihood of diabetes based on clinical parameters.")

if page == "🏠 Home":
    st.title("🏥 Welcome to HealthGuard")
    st.subheader("Diabetes Prediction System")
    
    st.markdown("""
    This application utilizes a Support Vector Machine (SVM) algorithm to predict whether a patient is likely to have diabetes based on various diagnostic measurements. 
    The model has been trained on the famous Pima Indians Diabetes Database.
    """)
    
    st.markdown("---")
    st.subheader("Dataset Statistics Overview")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df.columns) - 1}</div>
                <div class="metric-label">Features</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            diabetic_count = len(df[df['Outcome'] == 1])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{diabetic_count}</div>
                <div class="metric-label">Diabetic Cases</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            acc_percent = accuracy * 100 if accuracy else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc_percent:.1f}%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.info("👈 Please select a page from the sidebar to continue.")

elif page == "🔬 Diabetes Prediction":
    st.title("🔬 Diabetes Prediction using ML")
    st.markdown("Please enter the patient's diagnostic measurements below:")
    
    if model is not None and scaler is not None:
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=0, step=1, help='Number of times pregnant')
                glucose = st.number_input('Glucose', min_value=0, max_value=199, value=120, step=1, help='Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
                blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70, step=1, help='Diastolic blood pressure (mm Hg)')
                
            with col2:
                skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=20, step=1, help='Triceps skin fold thickness (mm)')
                insulin = st.number_input('Insulin', min_value=0, max_value=846, value=79, step=1, help='2-Hour serum insulin (mu U/ml)')
                bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0, step=0.1, help='Body mass index (weight in kg/(height in m)^2)')
                
            with col3:
                dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.5, step=0.001, help='Diabetes pedigree function')
                age = st.number_input('Age', min_value=21, max_value=81, value=33, step=1, help='Age (years)')
                
            submit_button = st.form_submit_button(label='Predict Result', use_container_width=True)
        
        if submit_button:
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            
            # Scale input data
            input_data_scaled = scaler.transform(input_data)
            
            # Predict
            with st.spinner('Analyzing...'):
                prediction = model.predict(input_data_scaled)
                prediction_proba = model.predict_proba(input_data_scaled)
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error('🚨 The person IS diabetic.')
                    st.warning(f'Confidence/Probability: {prediction_proba[0][1]*100:.2f}%')
                else:
                    st.success('✅ The person is NOT diabetic.')
                    st.info(f'Confidence/Probability: {prediction_proba[0][0]*100:.2f}%')
                
                with st.expander("View Contributing Factors (Feature Values)"):
                    st.write(f"- **Pregnancies**: {pregnancies}")
                    st.write(f"- **Glucose**: {glucose}")
                    st.write(f"- **Blood Pressure**: {blood_pressure}")
                    st.write(f"- **Skin Thickness**: {skin_thickness}")
                    st.write(f"- **Insulin**: {insulin}")
                    st.write(f"- **BMI**: {bmi}")
                    st.write(f"- **Diabetes Pedigree Function**: {dpf}")
                    st.write(f"- **Age**: {age}")
    else:
        st.error("Model could not be loaded. Please check the dataset.")

elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Outcome Distribution", "Correlation Heatmap", "Feature Distributions"])
        
        with tab1:
            st.write("Distribution of Diabetic vs Non-Diabetic Cases")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Outcome', data=df, palette='Set2', ax=ax)
            ax.set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'])
            ax.set_title('Outcome Distribution')
            st.pyplot(fig)
            
        with tab2:
            st.write("Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation between features')
            st.pyplot(fig)
            
        with tab3:
            st.write("Distributions of Key Features")
            feature_to_plot = st.selectbox("Select a feature to view its distribution", df.columns[:-1])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[feature_to_plot], kde=True, color='skyblue', ax=ax)
            ax.set_title(f'Distribution of {feature_to_plot}')
            st.pyplot(fig)
    else:
        st.error("Data could not be loaded.")

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    
    st.markdown("""
    ### About the Dataset
    This dataset is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**. 
    The objective of the dataset is to diagnostically predict whether a patient has diabetes based on certain diagnostic measurements included in the dataset.
    
    Several constraints were placed on the selection of these instances from a larger database. 
    In particular, all patients here are females at least 21 years old of Pima Indian heritage.
    
    ### About the Model
    The prediction engine uses a **Support Vector Machine (SVM)** classifier with a linear kernel. 
    The data is preprocessed using **StandardScaler** to ensure all features are on the same scale, which is crucial for SVM models.
    
    ### Disclaimer
    > **⚠️ Educational Purpose Only**  
    > This application is built for educational and demonstration purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    
    ### Credits
    Built with [Streamlit](https://streamlit.io/), [Scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), and [Seaborn](https://seaborn.pydata.org/).
    """)
