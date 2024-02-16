import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def visualize_data(training_data,test_data):
    """
    Gives out frequency of classes and visualizes it.
    """

    train_ratings = training_data["score"].value_counts().index.tolist()
    train_ratings_count = training_data["score"].value_counts().tolist()

    test_ratings = test_data["score"].value_counts().index.tolist()
    test_ratings_count = test_data["score"].value_counts().tolist()
    
    print("Training sentiment classes and per class frequency:")
    for i in range(len(train_ratings)):
        print(train_ratings[i], ": ", train_ratings_count[i])
     
    print("Test sentiment classes and per class frequency:")
    for i in range(len(test_ratings)):
        print(test_ratings[i], ": ", test_ratings_count[i])
    
    #visualize data in diagram using matplotlib
    plt.figure(facecolor='white')
    plt.title("Quantity per class - training data")
    plt.bar(x = train_ratings, height = train_ratings_count)
    plt.show()

    plt.figure(facecolor='white')
    plt.title("Quantity per class - testdata")
    plt.bar(x = test_ratings, height = test_ratings_count)
    plt.show()
    

def create_datasets(filepath):
    """
    Creates test and training data
    """
    
    with open(filepath, 'rb') as source:
        app_reviews = pd.read_csv(source)

    app_reviews['content'] = app_reviews['content'].astype('str')

    training_data, test_data = train_test_split(app_reviews, test_size=0.2, random_state=42, shuffle=True)
   
    print("Training data",len(training_data))
    print("Test data",len(test_data))
    
    return training_data,test_data


def create_baseline(training_data,test_data):
    """
    Create a baseline (a dummy model) that will guess on the most frequent class.
    Baselines are good to review before training so we know the trained model is better
    than one randomly generated.
    """
    
    dummy_clf = DummyClassifier(strategy="most_frequent")
   
    X = training_data['content']
    y = training_data['score']
    dummy_clf.fit(X, y)

    X_test = test_data['content']
    pred_mfc = dummy_clf.predict(X_test)
    
    #Create a confusion matrix to evaluate the dummy model.
    conf_mat = confusion_matrix(y_true = test_data.score,
                                y_pred = pred_mfc)

    labels = sorted(test_data['score'].unique())
    conf_mat_df = pd.DataFrame(conf_mat, index = [i for i in labels],
                               columns = [i for i in labels])
    print("confusion matrix of dummy model:")
    print(conf_mat_df)
    
    #Evaluate dummy model using recall, precision and F1-score
    class_report = classification_report(y_true=test_data.score,
                                        y_pred=pred_mfc,
                                        zero_division=0)

    # class_report_dict is used to control the result is correct
    class_report_dict = classification_report(y_true=test_data.score,
                                        y_pred=pred_mfc,
                                        zero_division=0,
                                        output_dict=True)
                        
    print("Classifier report:")
    print(class_report)
    

def Sentiment_analysis(training_data,test_data):
    """
    Created a sentiment analyser model using Naive Bayes.
    """
    
    vectorizer = CountVectorizer()
    clf = MultinomialNB()

    naive_bayes = Pipeline([('count_vect', vectorizer), ('multi_nb',clf)])

    X = training_data['content'].astype('str')
    y = training_data['score'].astype('int')

    naive_bayes.fit(X, y)

    X_test = test_data['content']
    pred_nb = naive_bayes.predict(X_test)

    #Evaluate model using recall, precision and F1-score
    class_report = classification_report(y_true=test_data['score'],
                                        y_pred=pred_nb)

    print("Classification report:")
    print(class_report)
    

def balance_data(training_data):
    """
    Balances out data by using the method undersampling.
    """
    
    train_ratings = training_data["score"].value_counts().index.tolist()
    train_ratings_count = training_data["score"].value_counts().tolist()
    
    min_class_freq = min(train_ratings_count)

    train_balanced = pd.DataFrame(columns=training_data.columns)
    for r in train_ratings:
        train_balanced = pd.concat([train_balanced, training_data.loc[training_data['score'] == r].sample(min_class_freq)])

    train_balanced.head()
    return train_balanced