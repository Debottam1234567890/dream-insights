# Dream Analyzer + Future Predictor

Welcome to **Dream Analyzer + Future Predictor** —  
An intelligent Python-based tool that analyzes your dreams and predicts your future moods, challenges, or opportunities based on dream psychology and AI!

---

## Features

### Dream Analysis
Predicts 10 key psychological factors from your dream description:
1. **Lucidity**
2. **Emotional Intensity**
3. **Realism**
4. **Fear Level**
5. **Joy Level**
6. **Control Over Dream**
7. **Symbolism Strength**
8. **Memory Recall After Waking**
9. **Strangeness**
10. **Vividness**

### Smart Mental Interpretation
Provides detailed, human-style explanations for each predicted factor.

### PCA Visualization
Projects your dream onto a 2D space alongside previous dreams, helping you see dream patterns visually.

### Dream Factor Bar Chart
Displays a beautiful bar chart showing how strongly each factor appeared in your dream.

### Real Life Prediction
Based on your dream’s emotional profile, the analyzer predicts a possible upcoming event or advice for your waking life!

---

## 🛠️ How It Works

### Data Preparation
- A CSV file (`dream_dataset.csv`) containing past dream descriptions and their annotated psychological factors is used to train the model.

### Model Training
1. Texts are vectorized using **TF-IDF**.
2. A **Random Forest Regressor** (wrapped in `MultiOutputRegressor`) is trained to predict the dream factors.

### Input Your Dream
- When you input a new dream description, the model analyzes it and predicts the values for each psychological factor.

### Interpretation and Visualization
- The system explains what each factor means.
- It projects your dream in a **PCA graph** and shows a **factor bar plot**.

### Bonus Prediction
- The analyzer generates a random but smart "fortune" about your near future based on your dream factors!

---

## Requirements

- **Python 3.8+**

### Libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install all dependencies via:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone or download the repository.
2. Make sure you have a `dream_dataset.csv` file with the following columns:
   - **Dream Description**
   - The 10 factor columns mentioned above.
3. Run the script:
   ```bash
   python dream_analyzer.py
   ```
4. Follow the prompt to enter your dream description.
5. Enjoy the analysis and get a glimpse into your future!

---

## Example Output

```
--- Input a New Dream ---
Enter dream description: I was flying over a beautiful ocean under a golden sky.

Predicted Factors:
Lucidity: 0.85
Emotional Intensity: 0.77
Realism: 0.30
...

--- Mental State Interpretation ---
High Lucidity: You may have had control and awareness inside the dream.
Strong Emotions: You experienced vivid emotions during the dream.
...

--- Real Life Prediction Based on Dream ---
Creativity will flow strongly soon — perfect time for projects!
```

---

## Project Structure

```
dream_analyzer/
├── dream_analyzer.py   # Main code
├── dream_dataset.csv   # Your dataset
└── README.md           # This file
```

---

## Future Ideas

- Make even multi-level future predictions (short-term and long-term).
- Add emotion classification using deep learning.
- Build a dream diary web app with this model.

---

## Why This Matters

Dreams reflect your subconscious mind.  
This project shows how **machine learning + human psychology** can combine to analyze dreams, visualize emotions, and even inspire your real-world actions!

*"Dreams are not just random — they are the subconscious stories we tell ourselves."*

---

## License

This project is licensed under the **MIT License**.  
Feel free to **edit, modify, and share**!  
See the [LICENSE](LICENSE) file for details.
