import joblib
import pandas as pd


model = joblib.load("models/model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

def predict_custom_house(next_to_river, nr_rooms, students_per_classroom, distance_to_town, pollution_level, poverty_level):
    input_data = pd.DataFrame([{
        'next_to_river': 1 if next_to_river else 0,
        'nr_rooms': nr_rooms,
        'students_per_classroom': students_per_classroom,
        'distance_to_town': distance_to_town,
        'pollution_level': 1 if pollution_level == 'high' else 0,
        'poverty_level': 1 if poverty_level == 'high' else 0
    }])[feature_names]

    return model.predict(input_data)[0]

if __name__ == "__main__":
    price = predict_custom_house(True, 8, 20, 5, 'high', 'low')
    print(f"預測價格: {price:.2f} 美元")
