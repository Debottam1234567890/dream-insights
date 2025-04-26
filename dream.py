import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

class DreamAnalyzer:
    def __init__(self, csv_path, target_column='Lucidity', test_size=0.2, random_state=42):
        """
        Initialize DreamAnalyzer with data loading, scaling, and optional TF-IDF.
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.scaler = None
        self.tfidf_vectorizer = None
        self.pca = PCA(n_components=2)
        self.feature_names = []
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df.dropna(inplace=True)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

        self.y = df[self.target_column]
        if 'Dream Description' in df.columns:
            self.has_text = True
            self.X_numeric = df.drop(columns=[self.target_column, 'Dream Description'])
            self.dream_texts = df['Dream Description']
        else:
            self.has_text = False
            self.X_numeric = df.drop(columns=[self.target_column])
            self.dream_texts = None

        # Scaling numeric features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X_numeric)

        # TF-IDF text features
        if self.has_text:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            X_text = self.tfidf_vectorizer.fit_transform(self.dream_texts).toarray()
            self.X_combined = np.hstack([X_scaled, X_text])
        else:
            self.X_combined = X_scaled

        # PCA
        self.pca_data = self.pca.fit_transform(self.X_combined)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_combined, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Feature names
        self.feature_names = list(self.X_numeric.columns)
        if self.has_text:
            self.feature_names += self.tfidf_vectorizer.get_feature_names_out().tolist()

    def plot_pca(self):
        """
        PCA plot of dreams.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_data[:, 0], y=self.pca_data[:, 1], hue=self.y, palette='viridis', s=100, alpha=0.7)
        plt.title('PCA Projection of Dream Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title=self.target_column)
        plt.grid(True)
        plt.show()

    def plot_feature_distribution(self):
        """
        Boxplot of numeric features.
        """
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.X_numeric, palette='coolwarm')
        plt.title('Feature Distribution Across Dreams')
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()

    def input_new_dream(self):
        """
        Allow user to input a new dream and see PCA and feature plots.
        """
        print("\n--- Input a New Dream ---")
        
        # Get dream description
        dream_description = ""
        if self.has_text:
            dream_description = input("Enter dream description (can be short): ")

        # Get numeric features
        input_values = []
        print("\nEnter values for the following numeric factors (press Enter to use 0):")
        for feature in list(self.X_numeric.columns):
            val = input(f"{feature}: ")
            if val.strip() == '':
                input_values.append(0.0)
            else:
                try:
                    input_values.append(float(val))
                except ValueError:
                    print("Invalid input. Defaulting to 0.0.")
                    input_values.append(0.0)

        # Scale numeric features
        scaled_numeric = self.scaler.transform([input_values])

        # Process dream description if available
        if self.has_text and self.tfidf_vectorizer:
            dream_tfidf = self.tfidf_vectorizer.transform([dream_description]).toarray()
            final_input = np.hstack([scaled_numeric, dream_tfidf])
        else:
            final_input = scaled_numeric

        # PCA projection
        projected = self.pca.transform(final_input)

        # Plot PCA
        plt.figure(figsize=(8, 5))
        plt.scatter(projected[:, 0], projected[:, 1], color='red', s=200, marker='*')
        plt.title('PCA Projection of Your New Dream')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

        # Plot feature barplot
        plt.figure(figsize=(14, 6))
        sns.barplot(x=list(self.X_numeric.columns), y=input_values, palette="mako")
        plt.title('Entered Dream Feature Values')
        plt.xticks(rotation=90)
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()

# Initialize
analyzer = DreamAnalyzer('dream_dataset.csv')

# Plot PCA of the dataset
analyzer.plot_pca()

# Plot feature distribution
analyzer.plot_feature_distribution()

# Input your new dream and see visualization
analyzer.input_new_dream()
