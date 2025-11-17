from flask import Flask, render_template, request
import pickle
import warnings
import re
import numpy as np

# Suppress version warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Clean and preprocess the input text"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def check_spam_keywords(text):
    """Check for common spam keywords and return score"""
    # High-confidence spam indicators (single occurrence is enough)
    strong_spam_keywords = [
        'lottery', 'winner', 'congratulations', 'won', 'prize', 'free money',
        'click here', 'claim', 'cash prize', 'million dollar', 'guaranteed',
        'risk free', 'no cost', 'act now', 'limited time offer', 'you have won',
        'claim your prize', 'winner selected', 'jackpot', 'big prize'
    ]
    
    # Medium-confidence spam indicators
    medium_spam_keywords = [
        'urgent', 'special promotion', 'exclusive offer', 'one time',
        'limited offer', 'buy now', 'order now', 'discount', 'save',
        'percent off', 'free trial', 'dollar', 'guaranteed', 'click now',
        'limited time', 'act fast', 'don\'t miss', 'once in a lifetime'
    ]
    
    text_lower = text.lower()
    
    # Check for strong indicators
    strong_count = sum(1 for keyword in strong_spam_keywords if keyword in text_lower)
    medium_count = sum(1 for keyword in medium_spam_keywords if keyword in text_lower)
    
    # Calculate spam score (0-1)
    spam_score = min(1.0, (strong_count * 0.5) + (medium_count * 0.2))
    
    # Return both boolean and score
    has_spam = strong_count >= 1 or medium_count >= 2
    
    return has_spam, spam_score

def calculate_text_features(text):
    """Calculate additional features that indicate spam"""
    text_lower = text.lower()
    
    # Count suspicious patterns
    exclamation_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    url_count = text.count('http') + text.count('www.')
    number_count = sum(1 for c in text if c.isdigit())
    
    # Calculate spam indicators
    spam_score = 0.0
    
    # Excessive exclamation marks
    if exclamation_count > 2:
        spam_score += 0.2
    
    # High caps ratio (shouting)
    if caps_ratio > 0.3:
        spam_score += 0.2
    
    # URLs present
    if url_count > 0:
        spam_score += 0.15
    
    # Many numbers (often in spam)
    if number_count > 5:
        spam_score += 0.1
    
    return min(1.0, spam_score)

try:
    cv = pickle.load(open("model/vectorizer.pkl", "rb"))
    clf = pickle.load(open("model/model.pkl", "rb"))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    cv = None
    clf = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", email_text="", label=None, confidence=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        userInput = request.form.get('email', '').strip()
        
        if not userInput:
            return render_template("index.html", error="Please enter an email to classify.", email_text="", label=None, confidence=None)
        
        if cv is None or clf is None:
            return render_template("index.html", error="Model not loaded. Please check model files.", email_text=userInput, label=None, confidence=None)
        
        # Preprocess the input text
        processed_text = preprocess_text(userInput)
        
        # Check for obvious spam keywords and calculate scores
        has_spam_keywords, keyword_spam_score = check_spam_keywords(userInput)
        feature_spam_score = calculate_text_features(userInput)
        
        # Combined spam indicator score
        combined_spam_score = max(keyword_spam_score, feature_spam_score)
        
        # Transform text to feature vector
        result = cv.transform([processed_text]).toarray()
        
        # Get prediction from model
        model_pred = clf.predict(result)
        model_pred = int(model_pred[0])
        
        # Get prediction probabilities
        spam_prob = 0.5
        not_spam_prob = 0.5
        all_probabilities = None
        
        try:
            if hasattr(clf, 'predict_proba'):
                all_probabilities = clf.predict_proba(result)[0]
                print(f"All probabilities: {all_probabilities}")
                
                # Handle multi-class models (4 classes in this case)
                if len(all_probabilities) == 4:
                    # For 4-class model, typically:
                    # Class 0,1 = Not Spam (ham)
                    # Class 2,3 = Spam
                    # Use the maximum probability class
                    max_prob_idx = int(np.argmax(all_probabilities))
                    max_prob = float(all_probabilities[max_prob_idx])
                    
                    # Map to binary: classes 2,3 are spam, 0,1 are not spam
                    if max_prob_idx >= 2:
                        # Spam class
                        pred = 1
                        spam_prob = max_prob
                        # Sum probabilities of spam classes (2 and 3)
                        spam_prob = float(all_probabilities[2] + all_probabilities[3])
                        not_spam_prob = float(all_probabilities[0] + all_probabilities[1])
                    else:
                        # Not spam class
                        pred = 0
                        not_spam_prob = max_prob
                        # Sum probabilities
                        spam_prob = float(all_probabilities[2] + all_probabilities[3])
                        not_spam_prob = float(all_probabilities[0] + all_probabilities[1])
                    
                    # Normalize to ensure they sum to 1
                    total = spam_prob + not_spam_prob
                    if total > 0:
                        spam_prob = spam_prob / total
                        not_spam_prob = not_spam_prob / total
                elif len(all_probabilities) == 2:
                    # Binary classification: [not_spam_prob, spam_prob]
                    not_spam_prob = float(all_probabilities[0])
                    spam_prob = float(all_probabilities[1])
                    pred = 1 if spam_prob > not_spam_prob else 0
                else:
                    # Single probability or unexpected format
                    max_prob = float(np.max(all_probabilities))
                    pred = model_pred
                    spam_prob = max_prob if pred == 1 else (1 - max_prob)
                    not_spam_prob = 1 - spam_prob
            else:
                # Fallback if no predict_proba
                pred = model_pred
                spam_prob = 0.75 if pred == 1 else 0.25
                not_spam_prob = 1 - spam_prob
        except Exception as e:
            print(f"Error getting confidence: {e}")
            import traceback
            traceback.print_exc()
            pred = model_pred
            spam_prob = 0.75 if pred == 1 else 0.25
            not_spam_prob = 1 - spam_prob
        
        # Apply rule-based correction with proper confidence calculation
        # Use combined spam score to adjust prediction and confidence
        keyword_confidence_boost = combined_spam_score * 0.25  # Scale boost based on spam score
        
        # Override prediction if spam indicators are strong
        if combined_spam_score > 0.6 and pred == 0:
            # Strong spam indicators but model says not spam - override
            print(f"Warning: Strong spam indicators (score: {combined_spam_score:.2f}) but model predicted not spam. Overriding to spam.")
            pred = 1
            # Use spam score to set confidence
            spam_prob = max(spam_prob, combined_spam_score)
            keyword_confidence_boost = 0.30  # Significant boost for override
        elif has_spam_keywords and pred == 1:
            # Model and keywords agree it's spam - boost confidence
            keyword_confidence_boost = 0.20 + (combined_spam_score * 0.15)
        elif combined_spam_score > 0.4:
            # Moderate spam indicators - moderate boost
            keyword_confidence_boost = 0.10
        
        # Calculate final confidence with multiple factors
        if pred == 1:
            # For spam prediction, combine model probability with spam indicators
            base_confidence = spam_prob * 100
            boosted_confidence = min(98.0, base_confidence + (keyword_confidence_boost * 100))
            
            # Use the higher of model confidence or spam indicator confidence
            indicator_confidence = combined_spam_score * 100
            confidence = max(boosted_confidence, indicator_confidence)
            
            # Ensure minimum confidence for spam with strong indicators
            if combined_spam_score > 0.6:
                confidence = max(confidence, 85.0)
            elif has_spam_keywords:
                confidence = max(confidence, 75.0)
        else:
            # For not spam, use model probability, reduce if spam indicators present
            base_confidence = not_spam_prob * 100
            if combined_spam_score > 0.3:
                # Reduce confidence if spam indicators present
                confidence = max(50.0, base_confidence - (combined_spam_score * 20))
            else:
                confidence = base_confidence
            
            # Ensure minimum confidence for not spam without indicators
            if combined_spam_score < 0.2:
                confidence = max(confidence, 60.0)
        
        print(f"Prediction result: {pred} (Spam=1, Not Spam=0)")
        print(f"Model spam probability: {spam_prob:.2%}, Not spam probability: {not_spam_prob:.2%}")
        print(f"Keyword spam score: {keyword_spam_score:.2f}, Feature spam score: {feature_spam_score:.2f}")
        print(f"Combined spam score: {combined_spam_score:.2f}")
        print(f"Final confidence: {confidence:.2f}%")
        print(f"Has spam keywords: {has_spam_keywords}")
        print(f"User input length: {len(userInput)} characters")
        
        return render_template("index.html", label=pred, confidence=round(confidence, 1), email_text=userInput, error=None)
    except Exception as e:
        print(f"Error in predict: {str(e)}")  # Debug output
        import traceback
        traceback.print_exc()
        return render_template("index.html", error=f"An error occurred: {str(e)}", email_text=request.form.get('email', ''), label=None, confidence=None)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Flask application...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)