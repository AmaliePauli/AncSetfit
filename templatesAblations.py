

DEV_DATASET_TO_METRIC = {
    "sst2": "accuracy",
    "imdb": "accuracy",
    "bbc-news": "accuracy",
    "student-question-categories": "accuracy",
}


DEV_TEMPLATES = {
    'sst2': 'The movie review is  ',
    "imdb": "The movie review is ",
    "bbc-news": "The topic is about ",
    "student-question-categories": "The subject is ",
}


DEV_TEMPLATES_A = {
    'sst2': "The movie is ",
    "imdb": "The movie is ",
}

DEV_TEMPLATES_Q = {
    'sst2': 'What is the movie review?  ',
    "imdb": "What is the movie review? ",
    "bbc-news": "What is topic in the news about? ",
    "student-question-categories": "What is the topic in the question about?  "
}

DEV_LABELS = {
    'sst2': ['negative','positive'],
    "imdb": ['negative', 'positive'],
    "bbc-news": ['tech', 'business', 'sport', 'entertainment', 'politics'],
    "student-question-categories": ['Biology', 'Chemistry', 'Maths', 'Physics'],
}

DEV_LABELS_A = {
    'sst2': ['bad', 'good'],
    "imdb": ['bad', 'good'],
}

