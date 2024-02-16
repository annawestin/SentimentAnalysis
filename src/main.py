from analyser import *

def main():
    
    doc = "../data/reviews.csv"
    
    #Create training and test datasets
    training_data, test_data = create_datasets(doc)

    #Create baseline to see if the trained model generates better results than a dummy one
    create_baseline(training_data,test_data)

    #Balance data
    train_balanced = balance_data(training_data)
    
    #Visualize data
    visualize_data(train_balanced,test_data)
    
    #Train model
    Sentiment_analysis(train_balanced,test_data)
    

#Run Main
#main()
if __name__ == "__main__":
    main() 