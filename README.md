# Gold Price Prediction: Deep-Learning

Gold has long been a favorable asset for hedging inflation. Particularly, it has been shining more after the COVID-19 breakout as most of all countries are heading for quantitative easing or QE again. Recently, the gold price hit the record high over $2000 per oz., and many analysts insist that there should be more room for price-increase in the next quarters. 

This brings us a question on how the machine would predict this trend, had we build a deep learning model with the historical gold prices. Would the model tell us a story of gold prices soaring like what they are like today?

In this project, we are going to read the historical gold prices from an API, building an RNN LSTM model for predicting the prices using time-series data. Our goal is to compare the prediction prices with the actual data to see how close the model can learn and predict the movement of gold prices.

## Data Preparation

In this section, we will retrieve the gold histroical prices from the London Bullion Market Association using Quandl API. The link is **[here](https://www.quandl.com/data/LBMA/GOLD-Gold-Price-London-Fixing)** and set the URL to retrieve the histroical prices of gold in `json` format. This can be done by importing `requests` and `json` from libraries and using `requests.get()`. We have daily data from Jan 2, 1968 to Aug 14, 2020 - nearly prices over 52 years!

## Data preprocessing

This is the most interesting and important part in dealing with any type of data. We have to go through from replacing the `null` or missing values. There were 6 different columns of prices; USD(AM), USD(PM), GBP(AM), GBP(PM), EURO(AM), and EURO(PM). Here, we are going to use the column of the USD(AM) for as feature and target column out of the six. By searching with `df.isnull().sum()`, we found out there were 1 missing values for the column. We filled the missing values with the previous price in the series.

Next, we will define a window size of 30 days for the data. The reason for setting a window(or rolling window) in time-series analysis is that we assume there is an unkown law so every next term of time-series is the function of *n* previous terms. For example, *r(t+1) = f(r(t),r(t-1),...,r(t-n-1))* and the influence of the rest time-series members is negligibly small. 
This will create arrays of 30 days value in time-series that will allow us to predict the next value based on the values in the window. Of course, we need to create the features set **X** and the target vector **y** as follows.


**X sample values:**   
[[35.18 35.16 35.14 35.14 35.14 35.14 35.15 35.17 35.18 35.18 35.19 35.2
  35.2  35.19 35.19 35.19 35.2  35.2  35.2  35.19 35.19 35.2  35.2  35.2
  35.19 35.19 35.2  35.19 35.2  35.19]   
 [35.16 35.14 35.14 35.14 35.14 35.15 35.17 35.18 35.18 35.19 35.2  35.2
  35.19 35.19 35.19 35.2  35.2  35.2  35.19 35.19 35.2  35.2  35.2  35.19
  35.19 35.2  35.19 35.2  35.19 35.19]   
 [35.14 35.14 35.14 35.14 35.15 35.17 35.18 35.18 35.19 35.2  35.2  35.19
  35.19 35.19 35.2  35.2  35.2  35.19 35.19 35.2  35.2  35.2  35.19 35.19
  35.2  35.19 35.2  35.19 35.19 35.2 ]]    

**y sample values:**   
[[35.19]   
 [35.2 ]   
 [35.2 ]]   

    
Since the time-series is order-sensitive, we have to avoid the dataset being randomized rather than using `train_test_split` function. We should manually create the training and testing sets using array slicing that separates 70% of the data for training and the remainder for testing. After then, we need to scale the dataset between 0 and 1 by using `MinMaxScaler`. The last part of preprocessing will be reshaping the features dataset. Because the LSTM API from Keras needs to receive the input data as a vertical vector, so that we have to reshape both sets, training and testing by using `reshape((X_train.shape[0], X_train.shape[1], 1))`.

## Build and Train the LSTM RNN

In this section, we will design a custom LSTM RNN in Keras and fit (train) it using the training data we defined.

For designing the structure of the RNN LSTM, we need to:

* Number of units per layer: `30` (same as the window size)
* Dropout fraction: `0.2` (20% of neurons will be randomly dropped on each epoch)
* Add three `LSTM` layers to your model, remember to add a `Dropout` layer after each `LSTM` layer, and to set `return_sequences=True` in the first two layers only.
* Add a `Dense` output layer with one unit.

After designing the model, we will complie the model using the `adam` optimizer(for gradient descent), and `mean_square_error` as loss function since the value we want to predict is continuous.

Now, we are going to train(fit) the model with the training data using 10 epochs and a `batch_size=90`. Since we are working with time-series data, do not forget to set `shuffle=False` as it's necessary to keep the sequential order of the data.

## Model Performance

In this section, we will evaluate the model using the test data. 

we will need to:

1. Evaluate the model using the `X_test` and `y_test` data.

2. Use the `X_test` data to make predictions.

3. Create a DataFrame of Real (`y_test`) vs. predicted values.

4. Plot the real vs. predicted values as a line chart.

The result we had was 0.0237909201, which is quite good accuracy. Then, we moved on to prediction using `X_test`, and used the `inverse_transform()` method of the scaler to decode the scaled testing and predicted values to their original scale.

Accordingly, we created a new dataframe called `stocks` with the datetime index from the `df` dataframe created previously, and with the columns of **Acutal** prices(testing data), and **Predicted** prices as follows.   

![](https://github.com/coolwonny/Gold-Price-Prediction/blob/master/Images/stocks_dataframe.png)   

Finally, we plotted the dataframe to see the overall trend.   
![plot](https://github.com/coolwonny/Gold-Price-Prediction/blob/master/Images/plot.png)

## Conclusion

With nearly 52 years of daily data, we trained the model with a dataset from Jan. 1968 to Nov. 2004. Then, tested it with the actual prices from Nov. 2004 up to the most recent of Aug 2020. Surprisingly, the model predicted quite precisely in the first few years(2004 to 2007). **More surprisingly, the model predicted the approximate trend for the next 13 years**, regardless of scalar gap between the acutal and predicted prices. How amazing!  
We learned from this project that how powerful an RNN-LSTM model could be, given a sufficient volume of time-series data.  
   
The entire process is documented in [Jupyter notebook](https://github.com/coolwonny/Gold-Price-Prediction/blob/master/gold_price_predict.ipynb).