# **StockPrediction**

Learning project to play with different machine learning models on time series (stock data)

### Install (Unix)

not setup as a full module yet

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **before api:**

1. Get data using the api entry point:

```bash
python -m app.get_data
```

This will pull the list of tickers from **data/tickers.json** To add tickers, update the active list in the json file


Run the model using the cli

```bash
python -m app.eval
```

This will run the logisitic regression model across the tickers listed in **data/tickers.json** and aggregate the data in the data/model_metrics folder

Regression model uses cross validation once per ticker to find the best inverse of the regularization strength, then runs walk forward 20 day rolling validation across the training and validation sets. Then runs a final test on the test data. 

Data is aggregated across tickers and stored as csv
