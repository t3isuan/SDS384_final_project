# Scientific machine learning final project: Coronavirus Tweets Text Classification

### Data source and data structure

Data source: [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification)

Data structure:
This dataset has already been split into train and test set. The train set has 44955 rows and 6 columns, and the test set has 7223 rows and 6 columns. The columns are as follows: `UserName`, `ScreenName`, `Location`, `TweetAt`, `OriginalTweet`, `Sentiment`. The `Sentiment` column is the target variable, and the other columns are the features. We will mainly use the `OriginalTweet` feature to train the model.

### Project description

The goal of this project is to predict the sentiment of tweets during the coronavirus pandemic. 

1. We first clean the data and perform exploratory data analysis to understand the data structure and the distribution of the data.
2. If possible, we will try to collect more data such as number of user's follower, likes number of the tweets, retweet number, and so on, to see if we can find some patterns in the data and improve the performance of the model.
3. Next, we will verctorize the text data and fit into various machine learning models to predict the sentiment of the tweets.
4. We also will try to use deep learning models like LSTM, Transformer and pretrained large language model to classify the sentiment of the tweets.
5. What's more we will compare the performance of the machine learning models and deep learning models and find out the best model for this task.
6. We will also try to get more tweets (maybe not related to COVID) from Twitter and use the model to predict the sentiment of the tweets.
7. According to models classification we then can find some patterns in the data and make some conclusions.

### Goal

The goal of this project is to accurate classify the sentiment of tweets during the coronavirus pandemic based on tweet content. We will also try to make classification on tweets in recent years not only COVID related tweets. And finally we We may be able to see changes in people's emotions at some key time points.