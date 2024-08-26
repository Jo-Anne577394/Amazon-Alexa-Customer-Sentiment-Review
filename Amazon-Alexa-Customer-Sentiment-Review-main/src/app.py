#libraries
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import contractions
import dash_table

# Load the model and vectorizer
model = joblib.load("./Amazon-Customer-Sentiment-Review/artifacts/model2.pkl")

#initialise dash app
app = dash.Dash(__name__)
server = app.server

# Sample DataFrame (replace this with your actual DataFrame)
df = pd.DataFrame({
    'rating': [5, 4, 3, 2, 1, 5],
    'date': ['2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05','2024-05-05'],
    'variation': ['Charcoal Fabric', 'Walnut Finish', 'Heather Gray Fabric', 'Sandstone Fabric', 'Oak Finish', 'Sandstone Fabric'],
    'verified_reviews': ['Love my Echo!', 'Sometimes while playing a game, you can answer a question correctly but Alexa says you got it wrong and answers the same as you.  I like being able to turn lights on and off while away from home.', 'Not what I expected', 'Disappointed', 'Worst purchase ever', 'MLG makes me tired!!']
})

# Define columns for DataTable
columns = [{"name": i, "id": i} for i in df.columns]

variation_options = [{'label': variation, 'value': variation.lower().replace(' ', '_')} for variation in df['variation'].unique()]


app.layout = html.Div(
    className='app-container', 
    style = {'background-color' : '#1d7874'},
    children=[
        html.H1('Amazon Alexa Review',
            style={ 'padding': '10px 20px',
                        'margin-top': '20px',
                        'font-family': 'calibri',
                        'font-weight': '600',
                        'font-size':'2em',
                        'color': '#071e22',
                        'width': '500px',
                        'text-align': 'center'}),
        html.Div( 
            className='content',
            style = {'background-color' : '#071e22',
                    'color': 'white',
                    'width':'200',
                    'font-family':'calibri',
                    'padding':'50px'},
            children=[
                html.Div([
                    html.Label('Rating', style={'display': 'block', 
                                                'margin-left': '25px',
                                                'font-size':'20px'}),
                    dcc.Slider(
                        id='rating-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=3,  # Set default value here
                        marks={i: str(i) for i in range(1, 6)},
                    )
                ], className='rating-container',
                style={'width':'500px'}), 
                html.Div([
                    html.Label('Date', style={'display': 'block',
                                                'margin-left': '25px',
                                                'font-size':'20px'}),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=None,
                        style={'margin-left': '25px'}
                    )
                ], className='date-container'), 
                html.Div([
                    html.Div([
                        html.Label('Variation', style={'display': 'block',
                                                        'margin-top':'20px',
                                                'margin-left': '25px',
                                                'color': 'white',
                                                'font-size':'20px'}),
                        dcc.Dropdown(
                            id='variation-dropdown',
                            options=variation_options,
                            value=None,
                            style={'margin-left': '13px', 'width': '465px', 'height': '50px', 'font-size':'22px', 'color': '#071e22'}
                        )
                    ], className='variation-container'),  
                    html.Div([
                        html.Label('Review', style={'display':'block',
                                                    'margin-top':'20px',
                                                    'margin-left': '25px',
                                                'font-size':'20px'}),
                        dcc.Textarea(
                            id='review-text',
                            value='',
                            style={'margin-left':'25px',
                                    'width':'460px',
                                    'height': '150px',
                                    'font-size':'large',
                                    'color':'#071e22'}
                        )
                    ], className='review-container'),  
                    html.Button('Submit', 
                                id='submit-val', 
                                n_clicks=0,
                                style={'background-color': '#1d7874',
                                        'color': 'white',
                                        'padding': '10px',
                                        'border': '1px solid transparent',
                                        'border-radius' : '10px',
                                        'margin-top':'10px',
                                        'margin-left':'25px'}),
                    html.Div(id='output-div')
                ], className='review-section'),
                html.Div([
                    html.H3('Amazon Reviews'),
                    html.Div(
                        dcc.Loading(
                            id='loading-table',
                            type='circle',
                            children=[
                                html.Div(
                                    dash_table.DataTable(
                                        id='review-table',
                                        columns=columns,
                                        data=df.to_dict('records'),
                                        page_size=10,  # Set the number of rows per page
                                        style_table={'overflowX': 'auto'},  # Enable horizontal scroll
                                        style_cell={'minWidth': '150px', 'width': '150px', 'maxWidth': '150px'},  # Set column width
                                        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},  # Set header style
                                        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},  # Set data style
                                        row_selectable='single',  # Allow only single row selection
                                        selected_rows=[]  # Initialize selected rows to empty list
                                    )
                                )
                            ]
                        )
                    )
                ], style={'overflowY': 'auto', 'height': '300px', 'margin-top': '20px', 'margin-left': '25px'})
            ]
        )
    ]
)

@app.callback(
    Output(component_id='rating-slider', component_property='value'),
    Output(component_id='date-picker', component_property='date'),
    Output(component_id='variation-dropdown', component_property='value'),
    Output(component_id='review-text', component_property='value'),
    Input(component_id='review-table', component_property='selected_rows'),
    State(component_id='review-table', component_property='data')
)
def update_inputs(selected_rows, data):
    if selected_rows:
        selected_row_index = selected_rows[0]
        selected_row = data[selected_row_index]
        rating = selected_row['rating']
        date = selected_row['date']
        variation = selected_row['variation']
        review = selected_row['verified_reviews']
        return rating, date, variation, review
    else:
        return 3, None, None, ''

@app.callback(
    Output(component_id='output-div', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    State('review-text', 'value')
)
def update_output(n_clicks, review_text):
    if n_clicks > 0 and review_text:  # Check if review text is not empty
        # Preprocess the review text
        processed_text = process(review_text)

        # Vectorize the processed text
        vectorized_text = vectorizer.transform([processed_text])

        # Make predictions
        predicted_sentiment = model.predict(vectorized_text)[0]

        # Interpret results
        if predicted_sentiment == -1:
            sentiment = "Negative Sentiment"
        elif predicted_sentiment == 0:
            sentiment = "Neutral Sentiment"
        else:
            sentiment = "Positive Sentiment"  

        return html.Div([
            html.H3("Predicted Sentiment:"),
            html.P(sentiment)
        ])
    else:
        return None  

def negate_sequence(text):
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = negated

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False

    return result

def process(text):
    # Convert text to lower case
    text = text.lower()
    
    # Remove all punctuation in text
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove HTML code or URL links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    
    # Fix abbreviated words
    text = contractions.fix(text)
    
    # Tokenize and handle negation
    tokens = negate_sequence(text)
    
    lemmatizer = WordNetLemmatizer()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    
    lemmatized_tokens = []
    
    for token in tokens:
        if token in stop_words:
            continue
        
        # Lemmatization
        lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
        
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

if __name__ == '__main__':
    app.run_server(port=8040, debug = True)
