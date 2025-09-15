# Dataset
movie_decision_data = {
    "weight_mood": {
        "happy": 2,
        "sad": 1,
        "adventurous": 3
    },
    "weight_time": {
        "short": 1,
        "long": 2
    },
    "weight_age": {
        "child": 1,
        "teen": 2,
        "adult": 3
    },
    "bias": 1
}

# Score-to-recommendation mapping (AI-like rule set)
score_to_recommendation = {
    range(0, 5): "🎥 Recommendation: Animated Movie (e.g. Toy Story)",
    range(5, 7): "🎥 Recommendation: Comedy or Romance (e.g. The Princess Diaries)",
    range(7, 9): "🎥 Recommendation: Action or Adventure (e.g. Indiana Jones)",
    range(9, 100): "🎥 Recommendation: Sci-Fi Thriller (e.g. Inception)"  # Catch-all for high scores
}

def recommend_movie(mood, time_available, age_group, data):
    weight_mood = data["weight_mood"]
    weight_time = data["weight_time"]
    weight_age = data["weight_age"]
    bias = data["bias"]

    # Calculate total score
    total_score = (
        weight_mood.get(mood.lower(), 0) +
        weight_time.get(time_available.lower(), 0) +
        weight_age.get(age_group.lower(), 0) +
        bias
    )

    # Rule-based recommendation (no if-else)
    recommendation = next(
        (message for score_range, message in score_to_recommendation.items() if total_score in score_range),
        "🎥 Recommendation: Unknown – try different inputs"
    )
    return recommendation

if __name__ == "__main__":
    mood_input = input("What's your mood? (happy/sad/adventurous): ")
    time_input = input("How much time do you have? (short/long): ")
    age_input = input("What's your age group? (child/teen/adult): ")

    recommendation = recommend_movie(mood_input, time_input, age_input, movie_decision_data)
    print(recommendation)
