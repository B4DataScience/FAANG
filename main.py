import pandas as pd
import pandas_datareader as pdr
import decimal
from decimal import *
from dateutil.relativedelta import relativedelta

import plotly.express as px
from dash import Dash, html, dcc, Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

getcontext().prec = 10

# %% Read Existing data
# Dataframe columns and their dtypes
data_columns = ["Date", "Close", "Volume", "Open", "High", "Low"]

# List of columns to crosscheck among various sources
compare_columns = ["Close", "Volume", "Open", "High", "Low"]

# Determines number of decimal places when rounding off decimals.
# 1.0000 means rounded value will have 4 decimal point precision
rounding_precision_exp = Decimal('1.0000')


# Convert string, float, int to decimal type
def convert_Decimal(f):
    if (isinstance(f, float)):
        f = "{:.5f}".format(f)
        # Using banker's rounding
        return Decimal(f).quantize(rounding_precision_exp, rounding=decimal.ROUND_HALF_EVEN)

    # if its a string or int
    # Using Decimal to be exact, rather than float. But floats are much faster :(
    # rounding to keep 4 decimal places using bankers rounding
    return Decimal(f).quantize(rounding_precision_exp, rounding=decimal.ROUND_HALF_EVEN)


# Dictionary of converter function for pandas read csv
convert_data_types = {
    "Close": convert_Decimal,
    "Open": convert_Decimal,
    "High": convert_Decimal,
    "Low": convert_Decimal
}
faang_tickers = {'FB': 'Facebook', 'AAPL': 'Apple', 'AMZN': 'Amazon', 'NFLX': 'Netflix', 'GOOGL': 'Google'}


class Stock:
    def __init__(self, ticker, name, data=None, source=None, startDate=None, endDate=None):
        self.ticker = ticker
        self.name = name
        self.source = source
        if source == 'File':
            self.data = pd.read_csv(
                f"Data/{ticker.lower()}.us.txt", index_col='Date',
                parse_dates=['Date'], converters=convert_data_types, usecols=data_columns)
            self.data.sort_index(inplace=True)
            self.startDate = self.data.index[0]
            self.endDate = self.data.index[-1]
            # a=Stock(ticker=ticker,name=name,source='Yahoo',startDate=self.startDate,endDate=self.endDate)
            # b=Stock(ticker=ticker,name=name,source='Quandl',startDate=self.startDate,endDate=self.endDate)
            # self.compareTwo(a=a,b=b)
            # self.compareTwo(a=self,b=b)

        elif source is not None:
            self.startDate = startDate
            self.endDate = endDate
            if source == 'Yahoo':
                self.data = pdr.get_data_yahoo(ticker, startDate, endDate)
            if source == 'Quandl':
                self.data = pdr.get_data_quandl(ticker, start=startDate, end=endDate)
            self.data.sort_index(inplace=True)
            self.update_dtype()

        else:
            self.source = 'Calculated'
            self.data = data
            self.startDate = startDate
            self.endDate = endDate

    # Update float types in data to Decimal
    def update_dtype(self):
        # Change dtypes for float values
        for col in compare_columns:
            if not col == "Volume":
                self.data[col] = self.data[col].apply(convert_Decimal)

    # compares two data
    def compareTwo(self, a, b):
        # Check difference in data
        for col in compare_columns:
            print(f"########### {a.ticker} {col} ###########")
            print(f"{a.source} Sum", a.data[col].sum())
            print(f"{b.source} sum", b.data[col].sum())
            print(f"{a.source} Max", a.data[col].max())
            print(f"{b.source} Max", b.data[col].max())
            print(f"{a.source} Min", a.data[col].min())
            print(f"{b.source} Min", b.data[col].min())


# Dictionary to keep tickers and their Stock data
faang_stocks = {ticker: Stock(ticker=ticker, name=faang_tickers.get(ticker), source='File') for ticker in faang_tickers}


# test_gf=pd.read_csv("~/Desktop/GF_FB.csv",index_col='Date', parse_dates=['Date'],converters={"Close": convert_Decimal})

# %% Quick checks
# dates_new_stock=[]
# for s in faang_stocks.values():
#     # print(f"### {s.ticker} ###\nStart={s.startDate}, End={s.endDate}")
#     # px.line(s.data, y='Close', title=s.ticker).show()
#     dates_new_stock.append(s.startDate)

# %% Functions required for index calculation

# When new stock is added for index calculation, resulting index shouldn't be changed drastically
# Divisor is updated to calibrate this. date is the day when new stock was added
def update_divisor(date):
    global faang_divisor
    # Opening stock prices on given date where these stocks were there in old divisor calculation
    recent_old = []
    for s in faang_stocks.values():
        # if stock was used in calculating old index or divisor
        if s.startDate < date:
            # NOTE: Considering new stock is added when market opens up
            recent_old.append(s.data.loc[date]["Open"])
    old_index = sum(recent_old) / faang_divisor

    recent = []
    for s in faang_stocks.values():
        # available stocks on a day
        if s.startDate <= date:
            recent.append(s.data.loc[date]["Open"])
    faang_divisor = sum(recent) / old_index


# Function to calculate Index, expects a dataframe row (summation of stock prices) to be calculated
def cal_index(row):
    global last_stocks_in_cal
    # If stock is added in index calculation, update divisor
    if row['N_Stocks'] > last_stocks_in_cal:
        last_stocks_in_cal += 1
        # rows were indexed with date in dataframe, apply function will return series with index as a name
        update_divisor(row.name)

    return row / faang_divisor


# %% Calculate FAANG stocks index fund

df_list = [s.data for s in faang_stocks.values()]
# dataframe with all faang stocks data combined
df = pd.concat(df_list)
df.drop('Volume', axis=1, inplace=True)

# Get the number of stocks available at the day
s_avil_day = pd.DataFrame(df.groupby(['Date']).size(), columns=["N_Stocks"])

# Sum faang stock's prices for the day. This summation is used to calculate the index.
faang_index = pd.DataFrame(df.groupby(['Date']).sum())

# join dataframe to have Summation and number of stocks types in one dataframe
faang_index = faang_index.join(s_avil_day)

# Algorithm to calculate index is similar to how DOW is calculated, taken from following link
# https://www.investopedia.com/articles/investing/082714/what-dow-means-and-why-we-calculate-it-way-we-do.asp
# TODO: use weighted average based on market capitalization like NASDAQ and S&P500
# NOTE: index is calculated from the beginning when there was only Apple stock,
# The index is updated as soon as new stock is available to add.
# Calculating fund index only when all 5 stocks are available, the index could be different.
# The calculation would be straightforward average in that case.

faang_index.sort_index(inplace=True)

# initial divisor is number of stock tickers (used to average out the summation)
faang_divisor = Decimal(str(faang_index.iloc[0]['N_Stocks']))
# number of stocks tickers in calculation
last_stocks_in_cal = faang_index.iloc[0]['N_Stocks']

# use existing summation to get index
faang_index = faang_index.apply(cal_index, axis=1)
del (faang_index['N_Stocks'])

# %% Plot data
# Create a collection to plot
stocks_data_dict = {}
data_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Name']
for s in faang_stocks.values():
    df = s.data
    df['Name'] = s.name
    stocks_data_dict[s.name] = df[data_columns]

faang_index['Name'] = 'FAANG Index'
stocks_data_dict['FAANG Index'] = faang_index

nasdaq_index = pdr.get_data_yahoo('^IXIC', faang_index.index[0], faang_index.index[-1])
nasdaq_index['Name'] = 'NASDAQ Index'
stocks_data_dict['NASDAQ Index'] = nasdaq_index[data_columns]

df_all = pd.concat(stocks_data_dict.values())
df_all.sort_index(inplace=True)

lowestDate = faang_index.index[0]
highestDate = faang_index.index[-1]
available_dates = faang_index.index

# stocks_view = faang_stocks
# stocks_view['FAANG'] = Stock(ticker='FAANG', name='FAANG Index', data=faang_index, startDate=lowestDate,
#                              endDate=highestDate)
# stocks_view['^IXIC'] = Stock(ticker='^IXIC', name='NASDAQ Index', source='Yahoo', startDate=lowestDate, endDate=highestDate)
# for s in stocks_view.values():
#     s.data.to_csv(f"View/Data/{s.ticker}.csv")
relative_deltas = {
    'oneWeek': relativedelta(days=7),
    'oneMonth': relativedelta(months=1),
    'oneYear': relativedelta(years=1),
    'threeYears': relativedelta(years=3),
    'fiveYears': relativedelta(years=5),
    'all': None
}

PRICECHART = 'Price'
PERCENTAGECHART = 'Percentage'
CANDLESTICCHART = 'Candlestic'

DAILYFREQ = "DAILY"
WEEKFREQ = "Weekly"
MONTHFREQ = "Monthly"
YEARFREQ = "Yearly"

# Color pallet taken from colorbrewer2.org
color_map={
    "Facebook":"#e41a1c",
    "Apple":"#377eb8",
    "Amazon":"#4daf4a",
    "Netflix":"#984ea3",
    "Google":"#ff7f00",
    "FAANG Index":"#a65628",
    "NASDAQ Index":"#ffff33"
}


# %% Initialize app its components

# takes row with close price and stock name, base value dictionary to calculate percent change from
def cal_perc_change(row, base_val_dict):
    base_val = base_val_dict.get(row['Name'])
    row['Close'] = round(((row['Close'] - base_val) / base_val) * 100, 2)
    return row


# Returns dataframe with given sampling freq, expected frequency's are daily, weekly and monthly
def sample_all(startDate, endDate, freq):
    if freq == DAILYFREQ:
        return df_all[startDate:endDate]
    elif freq == WEEKFREQ:
        delta = relative_deltas.get('oneWeek')
    elif freq == MONTHFREQ:
        delta = relative_deltas.get('oneMonth')
    elif freq == YEARFREQ:
        delta = relative_deltas.get('oneYear')
    sample_indexes = []
    date = startDate
    while date <= endDate:
        if date in available_dates:
            sample_indexes.append(date)
        date += delta
    return df_all.loc[sample_indexes]


def create_main(period, sampling, chart):
    # fig = go.Figure()
    # for s in stocks_view.values():
    #     fig = fig.add_trace(go.Scatter(x=s.data.index, y=s.data['Close'], name=s.name, mode='lines'))
    # # fig.update_layout(
    # #     margin=dict(l=0, r=0, t=0, b=0)
    # # )
    # return fig
    # Get Data
    delta = relative_deltas.get(period)
    if delta:
        startDate = highestDate - delta
    else:
        startDate = lowestDate
    df = sample_all(startDate, highestDate, sampling)
    # Build chart
    if chart == PRICECHART:
        fig = px.line(
            df,
            y='Close',
            color='Name',
            color_discrete_map=color_map,
            title="Price Change"
            # markers=True #Slows down when all data is plotted
        )
    elif chart == PERCENTAGECHART:
        df = df[['Close', 'Name']].copy()
        # df should be sorted since df_all was sorted and sampling goes from lower dates. Unless .loc changes the order
        stocks_in_df = df['Name'].unique().tolist()
        base_value_dict = {}
        for index, row in df.iterrows():
            sName = row['Name']
            if len(stocks_in_df) == 0:
                break
            elif sName in stocks_in_df:
                stocks_in_df.remove(sName)
                base_value_dict[sName] = row['Close']
        df = df.apply(lambda x: cal_perc_change(x, base_val_dict=base_value_dict), axis=1)
        fig = px.line(
            df,
            y='Close',
            color='Name',
            color_discrete_map=color_map,
            title='Percent Change'
            # markers=True
        )

    else:
        stocks_in_df = df['Name'].unique()
        groups = df.groupby('Name')
        fig = go.Figure()
        traces = []
        for stock in stocks_in_df:
            sdf = groups.get_group(stock)
            traces.append(
                go.Candlestick(
                    name=stock, x=sdf.index,
                    open=sdf['Open'], high=sdf['High'],
                    low=sdf['Low'], close=sdf['Close'],
                    # increasing_line_color= 'green', decreasing_line_color= 'red'

                )
            )
        fig.add_traces(traces)

        fig.update_layout(xaxis_rangeslider_visible=False,title="Price change Candlestick")
    # By Default make only Google visible, others can be made visible through clicking legends
    for trace in fig.data:
        if not trace.name=='Google':
            trace.visible='legendonly'
    return fig


def create_bar(period, sampling):
    delta = relative_deltas.get(period)
    if delta:
        startDate = highestDate - delta
    else:
        startDate = lowestDate
    df = sample_all(startDate, highestDate, sampling)
    fig = px.bar(
        df,
        y='Volume',
        barmode='group',
        color='Name',
        color_discrete_map=color_map,
        title='Volume Traded'
    )
    fig.update_traces(opacity=0.6,
                      marker_line_width=0,
                      selector=dict(type="bar"))
    # By Default make only Google visible, others can be made visible through clicking legends
    for trace in fig.data:
        if not trace.name=='Google':
            trace.visible='legendonly'
    return fig


app = Dash(__name__)

# Period Drop Down
period_dd = dcc.Dropdown(
    id='period_dd',
    options=[
        {'label': '1W', 'value': 'oneWeek'},
        {'label': '1M', 'value': 'oneMonth'},
        {'label': '1Y', 'value': 'oneYear'},
        {'label': '3Y', 'value': 'threeYears'},
        {'label': '5Y', 'value': 'fiveYears'},
        {'label': 'ALL', 'value': 'all'}
    ],
    value='oneWeek',
    searchable=False,
    clearable=False,
)
# sampling frequency drop down
sampling_dd = dcc.Dropdown([DAILYFREQ, WEEKFREQ, MONTHFREQ, YEARFREQ], DAILYFREQ, id='sampling_dd', searchable=False,
                           clearable=False)

# Chart type drop down
chart_dd = dcc.Dropdown([PRICECHART, PERCENTAGECHART, CANDLESTICCHART], PRICECHART, id='chart_dd', searchable=False,
                        clearable=False)

# TODO: Cross-filtering between charts, read https://dash.plotly.com/interactive-graphing
@app.callback(
    Output('Main', 'figure'),
    Input('period_dd', 'value'),
    Input('sampling_dd', 'value'),
    Input('chart_dd', 'value'),
)
def update_main(period, sampling, chart):
    return create_main(period, sampling, chart)


@app.callback(
    Output('Volume', 'figure'),
    Input('period_dd', 'value'),
    Input('sampling_dd', 'value')
)
def update_volume(period, sampling):
    return create_bar(period, sampling)


app.layout = html.Div(children=[
    html.Div(children=[html.Div(children=[html.Label("Period:",id='period_lab'), period_dd], id="period_main"),
                       html.Div(children=[html.Label("Basis(Sample):",id='sampling_lab'), sampling_dd], id="sampling_main"),
                       html.Div(children=[html.Label("Chart:",id='chart_lab'), chart_dd], id="chart_main")
                       ], id="options_main"
             ),
    dcc.Graph(id="Main"),
    dcc.Graph(id="Volume")
])

app.run_server(debug=False)

# %%
