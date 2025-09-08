#Dataset
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

 # Decision tree logic
    if total_score <= 4:
        return "🎥 Recommendation: Animated Movie (e.g. Toy Story)"
    elif total_score <= 6:
        return "🎥 Recommendation: Comedy or Romance (e.g. The Princess Diaries)"
    elif total_score <= 8:
        return "🎥 Recommendation: Action or Adventure (e.g. Indiana Jones)"
    else:
        return "🎥 Recommendation: Sci-Fi Thriller (e.g. Inception)"


if __name__ == "__main__":
    mood_input = input("What's your mood? (happy/sad/adventurous): ")
    time_input = input("How much time do you have? (short/long): ")
    age_input = input("What's your age group? (child/teen/adult): ")

    result = recommend_movie(mood_input, time_input, age_input, movie_decision_data)
    print(result)
