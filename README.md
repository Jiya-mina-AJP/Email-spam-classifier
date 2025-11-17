# Email Spam Classifier Web App

This project is a simple and interactive web application that classifies email text as either Spam or Not Spam. It uses a combination of machine learning and keyword-based rules to improve classification accuracy.

---

## Overview

Spam detection remains a key application of natural language processing. This project focuses on providing a practical, user-facing example of how spam filtering works by combining:

- A machine learning model (trained using TF-IDF and Logistic Regression)
- Rule-based detection of common spam patterns (keywords, formatting, etc.)

Through a Flask-based web interface, users can paste email content and get instant predictions with a confidence score.

---

## Features

- Classify emails in real-time through a web form
- Display the prediction outcome (Spam or Not Spam) with confidence
- Hybrid system: model prediction enhanced with keyword/pattern-based rules
- Ready for deployment using Gunicorn and Procfile

---

## Tech Stack

| Component    | Technology         |
|--------------|--------------------|
| Backend      | Python, Flask      |
| ML Model     | scikit-learn, NumPy|
| Frontend     | HTML, Jinja2       |
| Deployment   | Gunicorn, Procfile |

---

## Project Structure

email-spam-classifier/
├── app.py # Flask application
├── model/
│ ├── model.pkl # Trained machine learning model
│ └── vectorizer.pkl # TF-IDF vectorizer
├── templates/
│ └── index.html # Simple web UI
├── requirements.txt # Python dependencies
└── Procfile # Deployment configuration
