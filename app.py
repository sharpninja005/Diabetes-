import streamlit as st
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
import bcrypt
import plotly.graph_objects as go
from pathlib import Path
import re
import time
from streamlit_lottie import st_lottie
import json
import requests
import base64

# Configure Streamlit page
# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    /* Main page styling */
    .main {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Navigation bar */
    .navbar {
        background-color: rgba(0, 0, 0, 0.8);
        padding: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 1000;
    }
    
    .nav-links {
        display: flex;
        gap: 1rem;
    }
    
    .nav-link {
        color: white;
        text-decoration: none;
        font-size: 1.1rem;
        transition: color 0.3s;
    }
    
    .nav-link:hover {
        color: #1a73e8;
    }
    
    /* Card-like containers */
    .stForm {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Login page styling */
    .login-container {
        max-width: 400px;
        margin: 4rem auto;
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 4px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #1557b0;
        transform: translateY(-1px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-image: url("D:\Diabetes\medicine.jpg");
        background-size: cover;
        background-position: center;
        padding: 2rem;
    }
    
    /* Tour overlay */
    .tour-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
    }
    
    .tour-content {
        max-width: 600px;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        color: black;
    }

    /* Fix text input colors */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select,
    .stTextArea textarea {
        color: black !important;
        background-color: white !important;
    }
    
    /* Fix dropdown text color */
    .stSelectbox > div > div > div {
        color: black !important;
    }
    
    /* Fix chat input text color */
    .stChatInput input {
        color: black !important;
    }
    
    /* Fix contact page container */
    .content-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Additional card styling for contact page */
    .contact-card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    /* Add transparency to all containers */
    .stForm, .login-container, .gauge-container, div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Enhanced card styling */
    .card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Enhance form inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    /* Style the sidebar */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.95);
    }
    
    /* Style metrics containers */
    [data-testid="stMetricValue"] {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Chat message styling */
    .chat-message {
        background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .user-message {
        background-color: rgba(227, 242, 253, 0.9);
    }
    
    .bot-message {
        background-color: rgba(245, 245, 245, 0.9);
    }
    
    /* Loading animation background */
    .stMarkdown {
        background-color: transparent !important;
    }

    /* Label color fix */
    label {
        color: black !important;
    }

    /* Form field text color fix */
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #666666 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Paths
DB_PATH = Path("data/users.db")
MODEL_PATH = Path("models/Logistic_regression_Model_Diabetes.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# Create directories if they don't exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Remedies dataset
REMEDIES_DATA = {
    'low_risk': [
        "Maintain a balanced diet rich in fiber and low in processed sugars",
        "Exercise regularly - aim for 30 minutes of moderate activity daily",
        "Monitor your blood glucose levels periodically",
        "Stay hydrated with water instead of sugary beverages",
        "Get adequate sleep (7-9 hours per night)"
    ],
    'medium_risk': [
        "Consider consulting with a nutritionist for a personalized meal plan",
        "Increase physical activity to 45-60 minutes daily",
        "Monitor blood glucose levels more frequently",
        "Include more whole grains and vegetables in your diet",
        "Practice stress management techniques",
        "Consider taking supplements like alpha-lipoic acid under medical supervision"
    ],
    'high_risk': [
        "Schedule an appointment with an endocrinologist",
        "Start a strict blood glucose monitoring routine",
        "Follow a diabetes-specific meal plan",
        "Consider medical intervention under doctor's supervision",
        "Join a diabetes support group",
        "Practice portion control and meal timing",
        "Include diabetes-friendly exercises in your routine"
    ]
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email TEXT UNIQUE,
            created_at DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    except Exception:
        return False

def validate_password(password):
    checks = {
        'length': len(password) >= 8,
        'uppercase': bool(re.search(r'[A-Z]', password)),
        'lowercase': bool(re.search(r'[a-z]', password)),
        'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
        'number': bool(re.search(r'\d', password))
    }
    
    feedback = []
    if not checks['length']:
        feedback.append("Password must be at least 8 characters long")
    if not checks['uppercase']:
        feedback.append("Include at least one uppercase letter")
    if not checks['lowercase']:
        feedback.append("Include at least one lowercase letter")
    if not checks['special']:
        feedback.append("Include at least one special character")
    if not checks['number']:
        feedback.append("Include at least one number")
    
    return all(checks.values()), feedback

def load_model_and_scaler():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_gauge_chart(value, title, reference_range):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': reference_range},
            'bar': {'color': "#1a73e8"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': reference_range[1]
            }
        },
        title={'text': title}
    ))
    fig.update_layout(height=250)
    return fig

def validate_login(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=? OR email=?", (username, username))
        result = c.fetchone()
        conn.close()
        
        if result and verify_password(password, result[0]):
            return True
        return False
    except Exception:
        return False

def register_user(username, email, password):
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check existing user
        c.execute("SELECT 1 FROM users WHERE username=? OR email=?", (username, email))
        if c.fetchone():
            conn.close()
            return False
            
        # Insert new user
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                 (username, hashed_pw, email, datetime.now()))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False
    
    
    
    
    
    
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(""D:\Diabetes\medicine.jpg"");
             background-attachment: fixed;
             background-size: cover;
         }}
         
         /* Add transparency to all containers */
         .stForm, .login-container, .gauge-container, div[data-testid="stVerticalBlock"] > div {{
             background-color: rgba(255, 255, 255, 0.95) !important;
         }}
         
         /* Enhanced card styling */
         .card {{
             background-color: rgba(255, 255, 255, 0.95);
             border-radius: 10px;
             padding: 1.5rem;
             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
             margin: 1rem 0;
             backdrop-filter: blur(10px);
         }}
         
         /* Enhance form inputs */
         .stTextInput > div > div > input,
         .stSelectbox > div > div > div,
         .stNumberInput > div > div > input {{
             background-color: rgba(255, 255, 255, 0.9);
         }}
         
         /* Style the sidebar */
         .css-1d391kg {{
             background-color: rgba(255, 255, 255, 0.95);
         }}
         
         /* Style metrics containers */
         [data-testid="stMetricValue"] {{
             background-color: rgba(255, 255, 255, 0.9);
             padding: 1rem;
             border-radius: 8px;
         }}
         
         /* Chat message styling */
         .chat-message {{
             background-color: rgba(255, 255, 255, 0.9);
             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
         }}
         
         .user-message {{
             background-color: rgba(227, 242, 253, 0.9);
         }}
         
         .bot-message {{
             background-color: rgba(245, 245, 245, 0.9);
         }}
         
         /* Loading animation background */
         .stMarkdown {{
             background-color: transparent !important;
         }}
         </style>
         """,
         unsafe_allow_html=True)
     
    
    
    
    
def add_navigation():
    st.markdown("""
        <div class="navbar">
            <div class="nav-links">
                <a href="#" class="nav-link">Home</a>
                <a href="#about" class="nav-link">About Us</a>
                <a href="#contact" class="nav-link">Contact</a>
                <a href="#help" class="nav-link">Help</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    
def show_about_us():
    st.markdown("""
        <div id="about" class="stForm">
            <h2 style='color: #1a73e8; margin-bottom: 1.5rem;'>About Us</h2>
            <p style='font-size: 1.1rem; line-height: 1.6; color: #333;'>
                Welcome to our platform.<br><br>
                My name is Santhosh Kumar, and I am the developer of this website. This platform has been meticulously designed 
                to assist users in predicting their likelihood of having diabetes based on relevant health parameters.<br><br>
                Our goal is to provide a reliable, user-friendly tool that promotes early detection and awareness, empowering 
                individuals to make informed health decisions. By leveraging advanced algorithms, this website strives to offer 
                accurate and actionable insights to support your journey toward improved health.<br><br>
                Thank you for visiting, and we value your trust in our service. Please feel free to provide feedback to help 
                us improve further.
            </p>
        </div>
    """, unsafe_allow_html=True)






    

def show_login_page():
    add_bg_from_url()  # Add background image
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.markdown("""
            <div class="login-container card">
                <h1 style='text-align: center; color: black; margin-bottom: 2rem;'>
                    Welcome Back
                </h1>
                <p style='text-align: center; color: #5f6368; margin-bottom: 2rem;'>
                    Sign in to continue to Diabetes Prediction System
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.text_input("Email or Username", key="login_username")
            st.text_input("Password", type="password", key="login_password")
            
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                username = st.session_state.login_username
                password = st.session_state.login_password
                
                if validate_login(username, password):
                    st.success("Welcome back! Redirecting...")
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
def show_register_page():
    add_bg_from_url()  # Add background image

    # Inject custom CSS to change label color
    st.markdown("""
        <style>
        label {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="login-container card">
                <h1 style='text-align: center; color: black; margin-bottom: 2rem;'>
                    Create Account
                </h1>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            password2 = st.text_input("Confirm Password", type="password")
            
            if password:
                is_valid, feedback = validate_password(password)
                if not is_valid:
                    st.warning("Password Requirements:")
                    for msg in feedback:
                        st.warning(f"• {msg}")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Register", use_container_width=True)
            
            if submitted:
                if not all([username, email, password, password2]):
                    st.error("Please fill all fields")
                elif password != password2:
                    st.error("Passwords don't match")
                else:
                    is_valid, feedback = validate_password(password)
                    if not is_valid:
                        for msg in feedback:
                            st.error(msg)
                    elif register_user(username, email, password):
                        st.success("Registration successful! Please login.")
                        time.sleep(1)
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error("Username or email already exists")


def get_chatbot_response(query):
    query = query.lower()
    
    responses = {
        'hello': "Hello! How can I assist you with diabetes prevention today?",
        'diet': "A healthy diet for diabetes prevention should include: \n- Whole grains\n- Lean proteins\n- Vegetables\n- Limited processed sugars",
        'exercise': "Regular exercise is crucial! Aim for:\n- 30 minutes daily walking\n- Strength training 2-3 times/week\n- Yoga or stretching for flexibility",
        'symptoms': "Common diabetes symptoms include:\n- Frequent urination\n- Increased thirst\n- Unexplained weight loss\n- Fatigue\nPlease consult a doctor if you experience these.",
        'help': "I can help you with:\n- Diet advice\n- Exercise recommendations\n- Diabetes symptoms\n- General prevention tips"
    }
    
    for key, response in responses.items():
        if key in query:
            return response
    
    return "I'm here to help with diabetes-related questions. Try asking about diet, exercise, or symptoms!"

def show_chatbot():
    
    st.markdown(
    """
    <h1 style='text-align: center; color: black; background-color: white;'>Health Assistant</h1>
    """,
    unsafe_allow_html=True
)

    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_message = st.text_input("Type your message here (type 'exit' to close):", key="chat_input")
        submitted = st.form_submit_button("Send")
        
        if submitted and user_message:
            if user_message.lower() == 'exit':
                st.session_state.show_chat = False
                st.rerun()
            else:
                bot_response = get_chatbot_response(user_message)
                st.session_state.chat_history.append(("user", user_message))
                st.session_state.chat_history.append(("bot", bot_response))
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {message}
                </div>
            """, unsafe_allow_html=True)

def predict_diabetes(data, model, scaler):
    try:
        input_df = pd.DataFrame({
            'gender': [1 if data['gender'] == "Male" else 0],
            'age': [data['age']],
            'hypertension': [1 if data['hypertension'] == "Yes" else 0],
            'heart_disease': [1 if data['heart_disease'] == "Yes" else 0],
            'smoking_history': [0 if data['smoking_history'] == "Never" else 1 if data['smoking_history'] == "Current" else 2],
            'bmi': [data['bmi']],
            'HbA1c_level': [data['hba1c']],
            'blood_glucose_level': [data['glucose']]
        })
        
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[0]
        
        return prediction[0], probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Failed to load animation: {str(e)}")
        return None

# List of reliable Lottie animation URLs
LOTTIE_URLS = {
    "doctor": "https://lottie.host/1cbec0c9-88b6-439d-b820-e591b8c1f88d/Z9W0mhcKfD.json",  # Doctor animation
    "health_check": "https://lottie.host/1d1410af-3ea1-45da-a7e2-b83f45efe869/QeG2mfXaY2.json",  # Health checkup animation
    "heartbeat": "https://lottie.host/fc1d8fe7-7c1d-4b09-9d3b-d5dfe4556519/OB1gpxJEwz.json"  # Heartbeat animation
}

def show_prediction_page():
    add_bg_from_url()  # Add background image
    st.markdown("""
        <div class='card' style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1a73e8;'>Diabetes Risk Assessment</h1>
            <p style='color: #5f6368; font-size: 1.1rem;'>
                Enter your health metrics below for a comprehensive diabetes risk analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
   
    
    # Initialize chatbot state
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
    
    # Chat button
    if st.button("Open Health Assistant", key="chat_button"):
        st.session_state.show_chat = True
        st.rerun()
    
    # Show chatbot in full screen if enabled
    if st.session_state.show_chat:
        show_chatbot()
        return
    
    model, scaler = load_model_and_scaler()
    if not model or not scaler:
        st.error("Error: Could not load model. Please try again later.")
        return
    
    # Main prediction form with enhanced styling
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='input-group'>", unsafe_allow_html=True)
            smoking_history = st.selectbox("Smoking History", ["Never", "Current", "Former"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
            hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=9.0, value=5.0)
            glucose = st.number_input("Blood Glucose Level", min_value=70, max_value=300, value=100)
            st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            submitted = st.form_submit_button("Analyze Risk", use_container_width=True)
    
    # Process prediction and show results with enhanced styling
    if submitted:
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'hba1c': hba1c,
            'glucose': glucose
        }
        
        prediction, probabilities = predict_diabetes(input_data, model, scaler)
        
        if prediction is not None:
            # Show gauges in a container
            st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_gauge_chart(bmi, "BMI", [15, 40]), use_container_width=True)
            with col2:
                st.plotly_chart(create_gauge_chart(hba1c, "HbA1c", [4, 8]), use_container_width=True)
            with col3:
                st.plotly_chart(create_gauge_chart(glucose, "Glucose", [70, 200]), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show prediction with enhanced styling
            if prediction == 1:
                st.markdown("""
                    <div style='background-color: #ff4b4b; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;'>
                        <h3 style='color: white; margin: 0;'>⚠️ Based on the provided metrics, you may be at risk for diabetes.</h3>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background-color: #0bb04f; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;'>
                        <h3 style='color: white; margin: 0;'>✅ Based on the provided metrics, you appear to be at low risk for diabetes.</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show probabilities with enhanced styling
            st.markdown("""
                <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0;'>
                    <h3 style='color: #1a73e8; margin-bottom: 1rem;'>Risk Assessment</h3>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"No Diabetes Probability: {probabilities[0]*100:.1f}%")
                st.progress(probabilities[0])
            with col2:
                st.markdown(f"Diabetes Probability: {probabilities[1]*100:.1f}%")
                st.progress(probabilities[1])
            
            # Show remedies based on risk level
            risk_level = 'high_risk' if probabilities[1] > 0.7 else 'medium_risk' if probabilities[1] > 0.3 else 'low_risk'
            
            st.markdown("""
                <div style='background-color: black; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0;'>
                    <h3 style='color: #1a73e8; margin-bottom: 1rem;'>Recommended Prevention Steps</h3>
                """, unsafe_allow_html=True)
            
            for remedy in REMEDIES_DATA[risk_level]:
                st.markdown(f"<div style='margin: 2.45rem 0, color:black;'>• {remedy}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    add_bg_from_url()  # Add background image
    add_navigation()  # Add the navigation bar
    
    # Initialize session states
    if "page" not in st.session_state:
        st.session_state.page = "login"
   
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False

    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem; background: rgba(255, 255, 255, 0.95); border-radius: 10px;'>
                <h3 style='color: #1a73e8; margin-bottom: 1rem;'>Navigation</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.authenticated:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    st.session_state.page = "login"
            with col2:
                if st.button("Register", use_container_width=True):
                    st.session_state.page = "register"
        else:
            st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.95); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                    <p style='color: #1a73e8; font-size: 1.1rem; margin-bottom: 0.5rem;'>Welcome,</p>
                    <p style='color: #000000; font-weight: bold; font-size: 1.2rem;'>{st.session_state.username}!</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation buttons
            if st.button("Home", use_container_width=True):
                st.session_state.page = "prediction"
            
            if st.button("About Us", use_container_width=True):
                st.session_state.page = "about"
            
            if st.button("Contact", use_container_width=True):
                st.session_state.page = "contact"
            
            if st.button("Help", use_container_width=True):
                st.session_state.show_help = not st.session_state.show_help
            
            if st.button("Logout", use_container_width=True, key="logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.page = "login"
                st.rerun()

    # Main content area
    if st.session_state.authenticated:
        if st.session_state.page == "prediction":
            show_prediction_page()
        elif st.session_state.page == "about":
            show_about_page()
        elif st.session_state.page == "contact":
            show_contact_page()
        
        # Show help overlay if enabled
        if st.session_state.show_help:
            show_help_overlay()
            
    else:
        if st.session_state.page == "login":
            show_login_page()
        elif st.session_state.page == "register":
            show_register_page()

def show_about_page():
    st.markdown("""
        <div class='content-container'>
            <h1 style='color: #1a73e8; margin-bottom: 2rem;'>About Us</h1>
            <div class='card'>
                <p style='font-size: 1.1rem; line-height: 1.6; color: #000000;'>
                    Welcome to our platform.<br><br>
                    My name is Santhosh Kumar, and I am the developer of this website. This platform has been meticulously designed 
                    to assist users in predicting their likelihood of having diabetes based on relevant health parameters.<br><br>
                    Our goal is to provide a reliable, user-friendly tool that promotes early detection and awareness, empowering 
                    individuals to make informed health decisions. By leveraging advanced algorithms, this website strives to offer 
                    accurate and actionable insights to support your journey toward improved health.<br><br>
                    Thank you for visiting, and we value your trust in our service. Please feel free to provide feedback to help 
                    us improve further.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_contact_page():
    st.markdown("""
        <div class='content-container'>
            <h1 style='color: #1a73e8; text-align: center; margin-bottom: 2rem;'>Contact Us</h1>
            
            <div class='contact-card'>
                <h3 style='color: #1a73e8; margin-bottom: 1.5rem;'>Get in Touch</h3>
                <p style='color: #000000; font-size: 1.1rem; margin-bottom: 2rem;'>
                    Have questions or feedback? We'd love to hear from you. Feel free to reach out through any of the following channels:
                </p>
                
                <div style='margin-top: 2rem;'>
                    <div style='margin-bottom: 1rem;'>
                        <p style='color: #000000; font-size: 1.1rem;'>
                            <strong>📧 Email:</strong> contact@diabetesprediction.com
                        </p>
                    </div>
                    
                    <div style='margin-bottom: 1rem;'>
                        <p style='color: #000000; font-size: 1.1rem;'>
                            <strong>📞 Phone:</strong> +91 9500257398
                        </p>
                    </div>
                    
                    <div style='margin-bottom: 1rem;'>
                        <p style='color: #000000; font-size: 1.1rem;'>
                            <strong>📍 Address:</strong> 123 Health Street, Medical District, City
                        </p>
                    </div>
                    
                    <div style='margin-bottom: 1rem;'>
                        <p style='color: #000000; font-size: 1.1rem;'>
                            <strong>⏰ Business Hours:</strong> Monday - Friday, 9:00 AM - 5:00 PM EST
                        </p>
                    </div>
                </div>
            </div>
            
            <div class='contact-card' style='margin-top: 2rem;'>
                <h3 style='color: #1a73e8; margin-bottom: 1.5rem;'>Emergency Contact</h3>
                <p style='color: #000000; font-size: 1.1rem;'>
                    For medical emergencies, please dial your local emergency number or visit the nearest emergency room.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_help_overlay():
    with st.sidebar:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.95); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                <h4 style='color: #1a73e8; margin-bottom: 1rem;'>Help & Support</h4>
                <ul style='color: #000000; list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.5rem;'>• How to use the prediction tool</li>
                    <li style='margin-bottom: 0.5rem;'>• Understanding your results</li>
                    <li style='margin-bottom: 0.5rem;'>• FAQs</li>
                    <li style='margin-bottom: 0.5rem;'>• Contact support</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()