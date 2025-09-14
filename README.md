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

```jsx
python -m app.api
```

This will pull the list of tickers from **data/tickers.json** To add tickers, update the active list in the json file

1. Run a model
Run a model using the cli

```bash
python -m src.models.predictor --model basic_lr --ticker AAPL
```

This will print AUC ROC scores, the classification report, and confusion matrix information after running the basic logistic regression model

Available commands:

—models-list prints list of available models

—model [model] to chose which model to run

—ticker [ticker] to chose which ticker to run on, needs to have a processed data file ready