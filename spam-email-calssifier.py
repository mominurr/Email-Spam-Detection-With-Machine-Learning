# project name: Email Spam Detection With Machine Learning
# Author: Mominur Rahman
# Date: 27-11-2023
# Version: 1.0
# Description: This project is about Email Spam Detection With Machine Learning.
# GitHub Repo: https://github.com/mominurr/oibsip_task4
# LinkedIn: https://www.linkedin.com/in/mominur-rahman-145461203/

# Import necessary libraries
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


    

def mail_spam_detection():

    # Load the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # Display basic dataset information
    print("\nFirst 5 rows of the dataset: \n")
    print(df.head())
    
    print("\n\nLast 5 rows of the dataset: \n")
    print(df.tail())

    print("\n\nShape of the dataset: \n")
    print(df.shape)

    print("\n\nInformation about the dataset: \n")
    print(df.info())

    # Drop unnecessary columns
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1,inplace=True)

    # after dropping unnecessary columns
    print("\n\nAfter dropping unnecessary columns. Information about the dataset: \n")
    print(df.info())

    # Rename the columns
    df=df.rename(columns={'v1': 'label', 'v2': 'text'})

    # after renaming the columns
    print("\n\nAfter renaming the columns. 5 rows of the dataset: \n")
    print(df.head())


    print("\n\nNull or missing values in the dataset: \n")
    print(df.isnull().sum())

    
    print("\n\nUnique values in the dataset: \n")
    print(df.nunique())


    # remove duplicates if any
    df.drop_duplicates(inplace=True)

    # after removing duplicates
    print("\n\nAfter removing duplicates. shape of the dataset: \n")
    print(df.shape)
    
    
    # Convert labels to numerical values
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Visualize the distribution of spam and ham emails
    plt.figure(figsize=(10, 5))
    df['label'].value_counts().plot(kind='bar')
    # also show the plot ham is 0 and spam is 1
    plt.xticks([0, 1], ['Ham', 'Spam'])
    plt.title('Distribution of Spam and Ham Emails')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig('spam_distribution.png')
    plt.show()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['label'].values, test_size=0.2, random_state=42)

    # # Convert text data to numerical features using TF-IDF
    # tfidf_vectorizer = TfidfVectorizer()
    # X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Convert text data to numerical features using CountVectorizer
    count_vectorizer = CountVectorizer()
    X_train_tfidf = count_vectorizer.fit_transform(X_train)
    X_test_tfidf = count_vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    MODEL = MultinomialNB()
    MODEL.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = MODEL.predict(X_test_tfidf)

    

    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # print(f'\nAccuracy: {accuracy}')
    # print('\nClassification Report:')
    # print(classification_report(y_test, y_pred))
    # print('\nConfusion Matrix:')
    # print(confusion_matrix(y_test, y_pred))

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    
    # Plot the confusion matrix and include accuracy
    plt.figure(figsize=(12, 8))

    # Confusion Matrix Plot
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Display classification report
    plt.subplot(2, 2, 2)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
    plt.text(0.1, 0.1, report, {'fontsize': 12}, fontfamily='monospace')
    plt.axis('off')
    plt.title('Classification Report')

    # Display accuracy
    plt.subplot(2, 2, 3)
    plt.text(0.5, 0.5, f'Model Accuracy : {accuracy}', {'fontsize': 14}, ha='center', va='center')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("model_evaluation.png")
    plt.show()



    # Save the trained model
    joblib.dump(MODEL, 'spam_detection_model.pkl')

    # Save the vectorizer
    joblib.dump(count_vectorizer, 'count_vectorizer.pkl')





if __name__ == '__main__':
    print("\nWelcome to Email Spam Detection With Machine Learning Project\n")
    mail_spam_detection()







