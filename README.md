# Dream Analyzer - AI Dream Insights

Welcome to **Dream Analyzer**, your AI-powered dream analysis system combining psychology, neuroscience, and machine learning to interpret your dreams and provide insights into your subconscious mind.

**Privacy First:** No dream data is stored, neither locally nor globally. All analysis occurs only in-session.

---

## Features

### Comprehensive Dream Analysis

Predicts and analyzes 10 key psychological factors from your dream description:

1. Lucidity - Awareness within the dream
2. Emotional Intensity - Strength of feelings experienced
3. Realism - How realistic vs. surreal the dream was
4. Fear Level - Anxiety and threat perception
5. Joy Level - Positive emotions experienced
6. Control Over Dream - Degree of agency and influence
7. Symbolism Strength - Richness of symbolic content
8. Memory Recall After Waking - Clarity of dream memory
9. Strangeness - Bizarreness and illogical elements
10. Vividness - Clarity and detail of dream imagery

### Advanced Visualizations

* PCA Projection Graph for comparing dreams
* Dream Factor Bar Charts
* Pattern Recognition for recurring themes

### Intelligent Dream Interpretation

* Psychological analysis using Freudian, Jungian, cognitive, and modern neuroscience theories
* Symbol dictionary with extensive meanings
* Personalized, context-aware interpretations
* Multi-level analysis: literal, personal, psychological, archetypal, and existential

### Future Predictions & Guidance

* Mood forecasting based on dream factors
* Challenge warnings and potential obstacles
* Opportunities and areas of growth
* Practical actionable advice

### Mental State Interpretation

Provides explanations of:

* Current psychological state
* Unconscious processing
* Emotional regulation patterns
* Shadow work and personal growth

---

## How It Works

### Machine Learning Pipeline

1. Text vectorization (TF-IDF)
2. Multi-output prediction (Random Forest Regressor)
3. Pattern recognition via PCA
4. Ensemble intelligence combining statistical and psychological models

### Psychological Framework

* Neuroscience: REM sleep and memory consolidation
* Depth Psychology: Freudian and Jungian approaches
* Cognitive Science: dream formation theories
* Cross-Cultural: diverse interpretations

### Analysis Layers

1. Surface analysis (literal content)
2. Emotional profiling
3. Symbolic interpretation
4. Life connections
5. Future projection

---

## Requirements

**Python Version:** 3.8+

**Libraries:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask flask-cors
```

---

## How to Run

1. Clone or download the repository `dream-insights`
2. Ensure you have `dream_dataset.csv` with columns:

   * Dream Description
   * All 10 psychological factor columns
3. Start the API server:

```
python3 endpoints.py
```

4. Access the service at `http://localhost:5000`

---

## Sample CLI Output

```
dream-insights % python3 dream.py

--- Input a New Dream ---
Enter dream description: I was travelling through space.

Predicted Factors:
Lucidity: 0.74
Emotional Intensity: 0.75
Realism: 0.27
Fear Level: 0.38
Joy Level: 0.24
Control Over Dream: 0.44
Symbolism Strength: 0.58
Memory Recall After Waking: 0.51
Strangeness: 0.35
Vividness: 0.57

--- Mental State Interpretation ---
Lucidity: High Lucidity: You may have had control and awareness inside the dream.
Emotional Intensity: Strong Emotions: You experienced vivid emotions during the dream.
Realism: Low Realism: Your dream felt surreal or strange.
Fear Level: Low Fear: You likely felt calm or unafraid.
Joy Level: Low Joy: Your dream lacked joyful emotions.
Control Over Dream: Low Control: You were more of an observer in the dream.
Symbolism Strength: Strong Symbolism: Your dream contained rich, layered symbols.
Memory Recall After Waking: Strong Recall: You vividly remember the dream.
Strangeness: Low Strangeness: Your dream events made more logical sense.
Vividness: High Vividness: The dream imagery was sharp, colorful, and detailed.

--- Real Life Prediction Based on Dream ---
Creativity will flow strongly soon — perfect time for projects!
```

Note: The actual hosted app (https://dream-insights.onrender.com) has to be "waken up" at times due to Render (the hosting platform) requirements, but the actual loading of the app takes minimal time.
---

## Privacy & Ethics

* No dream data is stored locally or globally
* All analysis happens in-session only
* Interpretations are suggestions, not medical diagnoses
* Users are encouraged to consult professionals for trauma or distress

---

## Scientific Foundation

* Peer-reviewed sleep and dream research
* Psychological theories (Freud, Jung, cognitive, evolutionary)
* Modern neuroscience findings
* Evidence-based therapeutic techniques

---

## Contributing

* Add new dream symbols and cultural interpretations
* Enhance ML models and visualizations
* Multilingual support and accessibility

---

## License

MIT License — free to use, modify, and share responsibly

---

## About Dream Analyzer and Dream Bird

Dream Analyzer combines AI precision with depth psychology, respecting both science and the poetic nature of dreams. Your dreams remain private, secure, and insightful.

Dream Bird is the AI assistant behind Dream Analyzer. It is designed to help users explore their dreams, understand psychological patterns, and receive actionable insights. Dream Bird uses a comprehensive knowledge base of dream psychology, neuroscience, and cultural symbolism to provide nuanced interpretations and guidance.

May your dreams illuminate your path and guide your personal growth.

The website is available at: https://dream-insights.onrender.com
