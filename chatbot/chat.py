import nltk
from flask import Flask, render_template, request, session
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class ChatForm(FlaskForm):
    user_input = TextAreaField('Enter your text:', validators=[DataRequired()])
    submit = SubmitField('Send')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ChatForm()
    if 'chat_history' not in session:
        session['chat_history'] = []

    if form.validate_on_submit():
        user_input = form.user_input.data
        sentiment = analyze_sentiment(user_input)
        response = generate_response(user_input, sentiment)
        session['chat_history'].append(('User', user_input, {}))
        session['chat_history'].append(('Chatbot', response, sentiment))

    return render_template('index.html', form=form, chat_history=session['chat_history'])

def generate_response(user_input, sentiment):
    response = ""
    if sentiment['compound'] >= 0.05:
        response = "It sounds like you're feeling positive!"
    elif sentiment['compound'] <= -0.05:
        response = "It seems like you're feeling negative."
    else:
        response = "It looks like you're feeling neutral."

    response += " (Positive: {pos}, Neutral: {neu}, Negative: {neg}, Compound: {compound})".format(
        pos=sentiment['pos'],
        neu=sentiment['neu'],
        neg=sentiment['neg'],
        compound=sentiment['compound']
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)
