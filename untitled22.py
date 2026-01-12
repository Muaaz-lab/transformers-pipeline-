from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

reviews = [
    "This movie was absolutely fantastic, I loved every scene.",
    "The film was boring and too long, wasted my time.",
    "Amazing acting and brilliant direction.",
    "Worst movie I have ever seen in my life.",
    "The story was average but the music was great.",
    "I really enjoyed this movie, very entertaining.",
    "Terrible plot and poor acting.",
    "One of the best movies of this year!",
    "Not bad, but could have been much better.",
    "The movie was okay, nothing special.",
    "Excellent cinematography and visuals.",
    "I fell asleep halfway through the movie.",
    "The performances were outstanding.",
    "The script was weak and predictable.",
    "A masterpiece, truly inspiring film.",
    "Disappointing movie, had high expectations.",
    "Loved the background music and action scenes.",
    "The movie was confusing and poorly edited.",
    "Great story with emotional moments.",
    "I would not recommend this movie to anyone."
]

results = sentiment_model(reviews)

for review, result in zip(reviews, results):
    print("Review:", review)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 2))
    print("-" * 50)
