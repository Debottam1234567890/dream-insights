import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import random

class DreamAnalyzer:
    def __init__(self, csv_path, target_columns, test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.target_columns = target_columns
        self.test_size = test_size
        self.random_state = random_state

        self.scaler = None
        self.tfidf_vectorizer = None
        self.pca = PCA(n_components=2)
        self.model = None
        
        self.load_and_prepare_data()
        self.train_model()
    
    def load_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df.dropna(inplace=True)

        if 'Dream Description' not in df.columns:
            raise ValueError(f"'Dream Description' column not found.")

        self.X_text = df['Dream Description']
        self.y = df[self.target_columns]

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.X_vectorized = self.tfidf_vectorizer.fit_transform(self.X_text).toarray()

        self.pca_data = self.pca.fit_transform(self.X_vectorized)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_vectorized, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self):
        self.model = MultiOutputRegressor(RandomForestRegressor(random_state=self.random_state))
        self.model.fit(self.X_train, self.y_train)

    def plot_pca(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_data[:, 0], y=self.pca_data[:, 1], palette='viridis', s=100, alpha=0.7)
        plt.title('PCA Projection of Dream Descriptions')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

    def input_new_dream(self):
        print("\n--- Input a New Dream ---")
        
        dream_description = input("Enter dream description: ")

        dream_tfidf = self.tfidf_vectorizer.transform([dream_description]).toarray()

        predicted_factors = self.model.predict(dream_tfidf)[0]

        print("\nPredicted Factors:")
        for factor, value in zip(self.target_columns, predicted_factors):
            print(f"{factor}: {value:.2f}")

        # Mental state analysis with smarter interpretations
        print("\n--- Mental State Interpretation ---")
        threshold = 0.5

        interpretations = {
            'Lucidity': ("High Lucidity: You may have had control and awareness inside the dream.",
                         "Low Lucidity: The dream likely felt more automatic and uncontrolled."),
            'Emotional Intensity': ("Strong Emotions: You experienced vivid emotions during the dream.",
                                    "Mild Emotions: Emotional experiences were subtle or gentle."),
            'Realism': ("High Realism: Your dream closely mirrored waking life.",
                        "Low Realism: Your dream felt surreal or strange."),
            'Fear Level': ("High Fear: The dream involved anxiety, threats, or fear.",
                           "Low Fear: You likely felt calm or unafraid."),
            'Joy Level': ("High Joy: Your dream was filled with happiness and positivity.",
                          "Low Joy: Your dream lacked joyful emotions."),
            'Control Over Dream': ("Strong Control: You directed the flow of the dream events.",
                                   "Low Control: You were more of an observer in the dream."),
            'Symbolism Strength': ("Strong Symbolism: Your dream contained rich, layered symbols.",
                                   "Weak Symbolism: Your dream was more literal and straightforward."),
            'Memory Recall After Waking': ("Strong Recall: You vividly remember the dream.",
                                           "Weak Recall: Details of the dream may be fuzzy or lost."),
            'Strangeness': ("High Strangeness: Bizarre or impossible events filled your dream.",
                            "Low Strangeness: Your dream events made more logical sense."),
            'Vividness': ("High Vividness: The dream imagery was sharp, colorful, and detailed.",
                          "Low Vividness: The dream felt dull or faded.")
        }

        for factor, value in zip(self.target_columns, predicted_factors):
            high_message, low_message = interpretations.get(factor, ("High", "Low"))
            if value >= threshold:
                print(f"{factor}: {high_message}")
            else:
                print(f"{factor}: {low_message}")

        # PCA projection
        projected = self.pca.transform(dream_tfidf)

        plt.figure(figsize=(8, 5))
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], color='blue', alpha=0.3, label='Training Dreams')
        plt.scatter(projected[:, 0], projected[:, 1], color='red', s=200, marker='*', label='Your Dream')
        plt.title('PCA Projection of Your Dream')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot predicted factors
        plt.figure(figsize=(14, 6))
        sns.barplot(x=self.target_columns, y=predicted_factors, palette="mako")
        plt.title('Predicted Dream Factor Values')
        plt.xticks(rotation=90)
        plt.ylabel('Predicted Value')
        plt.grid(True)
        plt.show()

        # --------- New Part: Real Life Prediction ----------
        print("\n--- Real Life Prediction Based on Dream ---")

        prediction = self.generate_real_life_prediction(predicted_factors)
        print(prediction)

    def generate_real_life_prediction(self, factors):
        factor_dict = dict(zip(self.target_columns, factors))

        prediction_options = []

        if factor_dict.get('Joy Level', 0) > 0.7:
            prediction_options.append("A joyful surprise awaits you soon. 🌟")
        if factor_dict.get('Fear Level', 0) > 0.6:
            prediction_options.append("Be cautious — you might face a small challenge. 🛡️")
        if factor_dict.get('Lucidity', 0) > 0.7:
            prediction_options.append("You will soon have clarity over a confusing situation. 🔍")
        if factor_dict.get('Strangeness', 0) > 0.7:
            prediction_options.append("Expect unexpected encounters — strange but lucky! 🍀")
        if factor_dict.get('Symbolism Strength', 0) > 0.7:
            prediction_options.append("Hidden opportunities will reveal themselves to you. 🔮")
        if factor_dict.get('Control Over Dream', 0) > 0.7:
            prediction_options.append("You are about to take control over an important aspect of life. 🚀")
        if factor_dict.get('Memory Recall After Waking', 0) < 0.4:
            prediction_options.append("Be careful not to overlook small details this week. 🧠")
        if factor_dict.get('Vividness', 0) > 0.7:
            prediction_options.append("Creativity will flow strongly soon — perfect time for projects! 🎨")
        if not prediction_options:
            prediction_options.append("Life may continue steadily without major changes for now. 🌱")

        return random.choice(prediction_options)

# Target factor columns to predict
target_columns = [
    'Lucidity', 'Emotional Intensity', 'Realism', 'Fear Level', 'Joy Level',
    'Control Over Dream', 'Symbolism Strength', 'Memory Recall After Waking',
    'Strangeness', 'Vividness'
]

# Initialize
analyzer = DreamAnalyzer('dream_dataset.csv', target_columns=target_columns)

# Plot PCA of the dataset
analyzer.plot_pca()

# Input your new dream and see visualization + prediction
analyzer.input_new_dream()
