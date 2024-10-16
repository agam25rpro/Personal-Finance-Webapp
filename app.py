from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from statsmodels.tsa.holtwinters import Holt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Daily spending amount chart
    daily_spending = df.resample('D').sum()
    plt.figure(figsize=(10, 5))
    plt.bar(daily_spending.index, daily_spending['Amount'], color='skyblue')
    plt.title('Daily Spending Amount')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    daily_spending_chart = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.clf()

    # Forecasting using Holt's Linear Trend Model
    model = Holt(daily_spending['Amount'], exponential=False, damped_trend=True)
    model_fit = model.fit()
    forecast = model_fit.forecast(10)  # Forecast for the next 10 days

    # Creating a date range for the forecast
    forecast_index = pd.date_range(start=daily_spending.index[-1] + pd.Timedelta(days=1), periods=10)

    # Forecasting chart
    plt.figure(figsize=(10, 5))
    plt.plot(daily_spending.index, daily_spending['Amount'], label='Historical Spending', color='blue')
    plt.plot(forecast_index, forecast, label='Forecasted Spending', color='orange', marker='o')
    plt.title('Spending Forecast')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.axvline(x=daily_spending.index[-1], color='red', linestyle='--')  # Indicate the end of historical data
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    spending_forecast_chart = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()

    return render_template('index.html', daily_spending_chart=daily_spending_chart, spending_forecast_chart=spending_forecast_chart)

if __name__ == '__main__':
    app.run(debug=True)
