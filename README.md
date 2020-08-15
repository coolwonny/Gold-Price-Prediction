# Gold Price Prediction: Deep-Learning

Gold has long been a favorable asset for hedging inflation. Particularly, it has been shining more after the COVID-19 breakout as most of all countries are heading for quantitative easing or QE again. Recently, the gold price hit the record high over $2000 per oz., and many analysts insist that there should be more room for price-increase in the next quarters. 

This brings us a question on how the machine would predict this trend, had we build a deep learning model with the historical gold prices. Would the model tell us a story of gold prices soaring like what they are like today?

In this project, we are going to read the historical gold prices from an API, building an RNN LSTM model for predicting the prices using time-series data. Our goal is to compare the prediction prices with the actual data to see how close the model can learn and predict the movement of gold prices.

## Data Preparation

In this section, we will retrieve the gold histroical prices from the London Bullion Market Association using Quandl API. The link is **[here](https://www.quandl.com/data/LBMA/GOLD-Gold-Price-London-Fixing)** and set the URL to retrieve the histroical prices of gold in `json` format. This can be done by importing `requests` and `json` from libraries and using `requests.get()`. We have daily data from Jan 2, 1968 to Aug 14, 2020 - nearly prices over 52 years!

## Data preprocessing

This is the most interesting and important part in dealing with any type of data. We have to go through from replacing the `null` or missing values. There were 6 different columns of prices; USD(AM), USD(PM), GBP(AM), GBP(PM), EURO(AM), and EURO(PM). Here, we are going to use the column of the USD(AM) for as feature and target column out of the six. By searching with `df.isnull().sum()`, we found out there were 141 missing values for the column. We filled the missing values with the previous price in the series.

Next, we will define a window size of 30 days for the data. This will create arrays of 30 days value in time-series that will allow us  
