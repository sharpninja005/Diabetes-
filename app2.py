# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:31:30 2024

@author: Santhosh Kumar
"""

# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diabetes_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load the model and scaler
with open('models/Logistic_regression_Model_Diabetes.pkl', 'rb') as file:
    model = pickle.load(file)
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    prediction_result = db.Column(db.Float, nullable=False)
    gender = db.Column(db.Integer)
    age = db.Column(db.Integer)
    hypertension = db.Column(db.Integer)
    heart_disease = db.Column(db.Integer)
    smoking_history = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    hba1c_level = db.Column(db.Float)
    blood_glucose_level = db.Column(db.Float)

# Chatbot responses based on risk percentage
CHATBOT_RESPONSES = {
    (0, 20): [
        "Your diabetes risk appears to be low. Keep maintaining a healthy lifestyle!",
        "Great job on your health metrics! Continue your healthy habits.",
        "Your results look promising. Stay active and eat well!",
        "Low risk detected. Remember to have regular check-ups!"
    ],
    (20, 40): [
        "Your risk is moderate-low. Consider increasing physical activity.",
        "Watch your diet and exercise regularly to maintain good health.",
        "Some risk factors present. Regular monitoring would be beneficial.",
        "Consider consulting a nutritionist for dietary guidance."
    ],
    (40, 60): [
        "Moderate risk detected. Regular monitoring is recommended.",
        "Consider lifestyle modifications to reduce your risk factors.",
        "Schedule a consultation with your healthcare provider.",
        "Focus on improving your diet and exercise routine."
    ],
    (60, 80): [
        "Higher risk detected. Medical consultation is strongly advised.",
        "Important to monitor blood sugar levels regularly.",
        "Consider making significant lifestyle changes.",
        "Speak with your doctor about preventive measures."
    ],
    (80, 100): [
        "High risk detected. Immediate medical consultation recommended.",
        "Regular monitoring and lifestyle changes are crucial.",
        "Work closely with healthcare providers to manage risk factors.",
        "Important to develop a comprehensive health management plan."
    ]
}

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful!')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            input_data = pd.DataFrame({
                'gender': [int(request.form['gender'])],
                'age': [int(request.form['age'])],
                'hypertension': [int(request.form['hypertension'])],
                'heart_disease': [int(request.form['heart_disease'])],
                'smoking_history': [int(request.form['smoking_history'])],
                'bmi': [float(request.form['bmi'])],
                'HbA1c_level': [float(request.form['hba1c_level'])],
                'blood_glucose_level': [float(request.form['blood_glucose_level'])]
            })
            
            scaled_input = scaler.transform(input_data)
            prediction_prob = model.predict_proba(scaled_input)[0][1] * 100
            
            # Save prediction to database
            new_prediction = Prediction(
                user_id=session['user_id'],
                prediction_result=prediction_prob,
                gender=input_data['gender'].iloc[0],
                age=input_data['age'].iloc[0],
                hypertension=input_data['hypertension'].iloc[0],
                heart_disease=input_data['heart_disease'].iloc[0],
                smoking_history=input_data['smoking_history'].iloc[0],
                bmi=input_data['bmi'].iloc[0],
                hba1c_level=input_data['HbA1c_level'].iloc[0],
                blood_glucose_level=input_data['blood_glucose_level'].iloc[0]
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            # Get chatbot response
            for risk_range, responses in CHATBOT_RESPONSES.items():
                if risk_range[0] <= prediction_prob < risk_range[1]:
                    chatbot_response = np.random.choice(responses)
                    break
            
            return render_template('prediction_result.html', 
                                 prediction=prediction_prob, 
                                 chatbot_response=chatbot_response)
                                 
        except Exception as e:
            flash(f'Error during prediction: {str(e)}')
            return redirect(url_for('predict'))
            
    return render_template('predict.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.prediction_date.desc()).all()
    
    # Create visualization data
    dates = [p.prediction_date.strftime('%Y-%m-%d') for p in predictions]
    results = [p.prediction_result for p in predictions]
    
    # Create line chart
    fig = px.line(x=dates, y=results, 
                  title='Diabetes Risk History',
                  labels={'x': 'Date', 'y': 'Risk Percentage'})
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('history.html', 
                         predictions=predictions, 
                         graph=graphJSON)

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    user_predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    
    # Prepare data for visualizations
    data = pd.DataFrame([{
        'age': p.age,
        'bmi': p.bmi,
        'blood_glucose': p.blood_glucose_level,
        'hba1c': p.hba1c_level,
        'risk': p.prediction_result
    } for p in user_predictions])
    
    # Create various plots
    scatter_fig = px.scatter(data, x='bmi', y='blood_glucose', 
                           color='risk', title='BMI vs Blood Glucose')
    
    box_fig = px.box(data, y=['blood_glucose', 'hba1c'], 
                     title='Distribution of Key Metrics')
    
    # Convert plots to JSON for rendering
    graphs = {
        'scatter': json.dumps(scatter_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'box': json.dumps(box_fig, cls=plotly.utils.PlotlyJSONEncoder)
    }
    
    return render_template('analytics.html', graphs=graphs)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0',port=8080,debug=True)