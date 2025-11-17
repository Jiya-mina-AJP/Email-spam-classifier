# Email Spam Classification Flask App

A web application that classifies emails as spam or not spam using a machine learning model.

## Features

- Simple web interface for email spam classification
- Real-time prediction using pre-trained ML model
- Clean and intuitive UI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter or paste an email message in the text area
2. Click "Predict" to classify the email
3. The result will be displayed as either "Spam" (red) or "Not Spam" (green)

## Project Structure

```
email-spam-classification-flask/
├── app.py                 # Flask application
├── model/
│   ├── model.pkl         # Trained ML model
│   └── vectorizer.pkl    # Text vectorizer
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
└── Procfile             # Deployment configuration
```

## Deployment

The project includes a `Procfile` for deployment on platforms like Heroku. Make sure `gunicorn` is installed (included in requirements.txt).

