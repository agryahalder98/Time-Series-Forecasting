# Time-Series-Forecasting
This repo includes neural network based time series forecasting for the kaggle's Store Sales - Time Series Forecasting competition. The competition consists on forecasting unit sales for thousands of items sold at convenience stores (Favorita from Ecuador). The training dataset includes: date, store and product information, whether an item was promoted or not, and the sales numbers. The dataset contains additional supplementary information that may contribute to the machine learning algorithm. This time series competition requires leveraging external information, as well as the sales numbers from several items. Please, visit the official competition webpage to get a more detailed description of the dataset and access to said dataset. URL: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview

The test set is blind, you do not have sales numbers for the test set. To obtain test metrics for ANY model the competitors have to upload to Kaggle a submission.csv file with the format of the contest.

The dataset is opensourced and available to download at kaggle.com.

The EDA files are used to analyse the dataset from different crucial aspects. Simply run it by ./{filename}.py to generate all EDA analysis. The generated files are stores automatically in the prescribed folder. To play with EDA, use ipynb file.

For the baseline, ARIMA is chosen and the code is given in ARIMA.ipynb. 

For the neural network, an combination of CNN and LSTM architecture is used, namely CNN-LSTM, to make the prediction.
