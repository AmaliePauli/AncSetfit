
TEMPLATES = {
    'sst5': 'The movie is  ',
    'CR': 'The movie is  ',
    'emotion': 'The emotion is ',
    'ag_news':  "The topic is about ",
    'yelp_review_full': 'The experience was ',
    'yahoo_answers_topics': 'The question is about '
    
}

LABELS= {
    'sst5': ['terrible', 'bad', 'okay', 'good', 'great'], 
    'emotion': ['sadness','joy','love','anger','fear','surprise'],
    'CR': ['bad','good'],
    'ag_news': ['World', 'Sports', 'Business', 'Sci/Tech'], 
    'yelp_review_full': ['terrible', 'bad', 'okay', 'good', 'great'], 
    'yahoo_answers_topics': ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music', 'Family & Relationships', 'Politics & Government']
}

DATASET_TO_METRIC = {
    "sst5": "accuracy",
    "emotion": "accuracy",
    'CR': "accuracy",
    'ag_news': "accuracy",
    'yelp_review_full': 'accuracy',
    'yahoo_answers_topics': 'accuracy',
}

DEV_DATASET_TO_METRIC = {
    "sst2": "accuracy",
    "imdb": "accuracy",
    "bbc-news": "accuracy",
    "student-question-categories": "accuracy",
    "TREC-QC": "accuracy",
    'toxic_conversations': "matthews_correlation",
}


DEV_TEMPLATES = {
    'sst2': 'The movie review is  ',
    "imdb": "The movie review is ",
    "bbc-news": "The topic is about ",
    "student-question-categories": "The subject is ",
    "TREC-QC": "The question is regarding ",
    'toxic_conversations': "The tone in the conversation was ",
}

DEV_LABELS = {
    'sst2': ['negative','positive'],
    "imdb": ['negative', 'positive'],
    'toxic_conversations': ['fine', 'toxic'],
    "bbc-news": ['tech', 'business', 'sport', 'entertainment', 'politics'],
    "student-question-categories": ['Biology', 'Chemistry', 'Maths', 'Physics'],
    "TREC-QC": ['manner of an action',
 'inventions, books and other creative pieces',
 'animals',
 'expression abbreviated',
 'an individual',
 'a group or organization of persons',
 'title of a person',
 'definition of something',
 'dates',
 'reasons',
 'events',
 'states',
 'description of something',
 'number of something',
 'other entities',
 'letters like a-z',
 'other locations',
 'religions',
 'food',
 'countries',
 'colors',
 'equivalent terms',
 'cities',
 'organs of body',
 'diseases and medicine',
 'mountains',
 'prices',
 'products',
 'the lasting time of something',
 'elements and substances',
 'sports',
 'plants',
 'techniques and methods',
 'size, area and volume',
 'description of a person',
 'musical instrument',
 'abbreviation',
 'other numbers',
 'speed',
 'words with a special property',
 'languages',
 'fractions',
 'postcodes or other codes',
 'linear measures',
 'temperature',
 'symbols and signs',
 'ranks',
 'vehicles',
 'weight',
 'currency names']
}

MULTI_TEMPLATE = {
    'go_emotions': 'The emotions are ',
    'abstract': 'The paper is in the field of ',
    'semeval2018task1': 'The emotions are ',
}

MULTI_LABELS= {
     'go_emotions': ['admiration','amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'],
     'abstract': ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'],
    'semeval2018task1': ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
}

