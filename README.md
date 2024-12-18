# Email Spam Detection With Machine Learning

## Project Overview
This project aims to build a spam detection model using machine learning. The dataset used contains text messages labeled as "ham" (non-spam) or "spam." The project involves data analysis, visualization, and the implementation of a Naive Bayes classifier for spam detection.

## Data Analysis
- Loaded and inspected the dataset.
- Handled missing values and unnecessary columns.
- Renamed columns for clarity.
- Removed duplicates from the dataset.
- Converted labels to numerical values.

## Data Visualization
- Plotted the distribution of spam and ham emails.
- Visualized the confusion matrix, classification report, and model accuracy.

## Script Details
The script `Email_Spam_Detection_ML.ipynb` performs the following tasks:
1. Imports necessary libraries.
2. Loads and analyzes the dataset.
3. Preprocesses the data, including dropping unnecessary columns, renaming columns, handling missing values, and removing duplicates.
4. Converts text data to numerical features using CountVectorizer.
5. Trains a Naive Bayes classifier (MultinomialNB) for spam detection.
6. Evaluates the model, generates a confusion matrix, and visualizes the results.
7. Saves the trained model and vectorizer.



**Model Use**: After training, this model is used for prediction.

## Live Project
The application is hosted live on Streamlit, making it easily accessible for users to input email text and get real-time predictions.

**The live project can be accessed at the following URL:** [Live Web App - Email Spam Detection](https://email-spam-detections.streamlit.app)

## Requirements
Ensure you have the following libraries installed to run the script:

- pandas
- joblib
- seaborn
- matplotlib
- scikit-learn
- flask
- streamlit

Install the required libraries using:

    pip install -r requirements.txt
    
## Usage(In your local machine)
To use this project, follow these steps:
1. Ensure you have Python installed on your machine.
2. **For Training:**
   - Clone the repository: `git clone https://github.com/mominurr/Email-Spam-Detection-With-Machine-Learning.git`
   - Install the required libraries: `pip install -r requirements.txt`
3. **For Prediction via Flask:**
   - Run the script: `python flax_app.py`

## Conclusion
This spam detection project successfully builds and visualizes a machine learning model for identifying spam messages. The Multinomial Naive Bayes classifier demonstrates effective performance, with insights provided by visualizations such as the distribution of spam and ham emails and the confusion matrix. The trained model and CountVectorizer are saved for future use. The project serves as a foundation for spam detection tasks, and contributions or extensions are welcome. Thank you for exploring this spam detection project!

##Author:
[Mominur Rahman](https://github.com/mominurr)
