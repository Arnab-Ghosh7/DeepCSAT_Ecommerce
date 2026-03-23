# DeepCSAT: E-Commerce CSAT Score Predictor 🎯

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <br>
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
</div>

<br>

---



## 📖 Overview

In modern e-commerce, anticipating customer dissatisfaction allows proactive resolution. DeepCSAT takes various interaction parameters (such as the communication channel, issue category, customer remarks, response time, and agent details) and feeds them into a trained Deep Learning model to predict a CSAT score on a scale of 1 to 5.

**Key Highlights:**
- Trained on **85.9K** interaction records.
- Employs a **3-Layer ANN** tuned via RandomizedSearchCV.
- Handles natural language remarks using **TF-IDF vectorization**.
- Provides real-time probability distributions for all 5 score classes.

## ✨ Features

- **Live Predictions:** Instantly outputs a predicted CSAT score along with full class probabilities.
- **NLP Integration:** Parses and vectorizes optional customer remarks to deeply understand customer sentiment.
- **Robust Preprocessing:** Automatically handles text cleaning, log-transformations for skewed numerical features, and label encoding for categorical variables.
- **Beautiful Interface:** A modern web UI with real-time probability bars, animated SVGs, and interactive metric displays.
- **Version Compatibility:** Includes a custom Unpickler module that silently handles library version mismatches between `numpy` and `scikit-learn`.

## 🛠️ Technologies Used

- **Machine Learning & NLP:** `scikit-learn` (Label Encoding, TF-IDF, StandardScaler), `numpy`, `pandas`
- **Deep Learning:** `scikit-learn` Neural Network (MLPClassifier)
- **Backend:** Python, `Flask`
- **Frontend:** HTML5, CSS3 (Custom Variables, Grid/Flexbox), JavaScript (Fetch API)

## 🧠 How the Model Works
1. **Categorical Features:** Inputs (like Channel Name, Shift, Category, Agent) are label-encoded using encoders fitted during training to ensure an exact mapping.
2. **Text Processing:** Customer remarks are cleaned (stopwords/contractions removed) and vectorized via TF-IDF (extracting the top 200 bigram features).
3. **Numerical Features:** Metrics such as Response Time and Item Price are log-transformed and scaled using `StandardScaler`.
4. **Prediction:** The processed features are horizontally concatenated and passed through the 3-layer ANN, predicting the likelihood of each CSAT score from 1 (Very Dissatisfied) to 5 (Very Satisfied).


## 📂 Project Structure

```text
DeepCSAT_Ecommerce/
│
├── app.py                            # Flask application and inference API
├── deepcsat_model.pkl                # Pre-trained ANN model and preprocessing artifacts
├── DeepCSAT_Ecommerce_Final.ipynb    # Jupyter Notebook tracking EDA, training, and model tuning
└── templates/
    └── index.html                    # Frontend UI design and client-side logic
```

## 🚀 Setup & Installation

**Prerequisites:** Python 3.8+ is recommended. 

1. **Clone the repository:**
   ```bash
   https://github.com/Arnab-Ghosh7/DeepCSAT_Ecommerce
   cd DeepCSAT_Ecommerce
   ```

2. **Install the dependencies:**
   Make sure to install the standard data science and web packages:
   ```bash
   pip install Flask numpy pandas scikit-learn
   ```
   *(Note: The model pickling expects compatible `scikit-learn` and `numpy` versions. If you encounter an unpickling error upon startup, you can re-run the provided notebook `DeepCSAT_Ecommerce_Final.ipynb` on your machine to regenerate the `.pkl` file).*

3. **Run the Application:**
   ```bash
   python app.py
   ```

4. **Access the Web App:**
   Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

## 💻 Usage

1. Open the application in your browser.
2. Fill out the **Channel & Interaction** specifics (eg., Outcall, Product Queries).
3. Optionally, add **Customer Remarks** to capture qualitative sentiment.
4. Input **Agent Details** and **Interaction Metrics** (Response Time, Item Price).
5. Click **Predict CSAT Score**. The UI will smoothly transition to display the most likely score and the granular probability distribution across all 5 classes.
