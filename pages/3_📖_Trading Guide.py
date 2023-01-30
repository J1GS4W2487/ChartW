import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import warnings
from PIL import Image
import matplotlib.dates as mpdates
from plotly.offline import iplot
import cufflinks as cf
import pandas as pd
from datetime import datetime
import pandas_datareader as data
import streamlit as st
import plotly.graph_objects as go
# Import necessary libraries
from PIL import Image
import pandas as pd
import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp
from PIL import Image

from dateutil.relativedelta import relativedelta

import datetime

st.header("Learning to Trade 💡")
st.write("#")

st.subheader('_Stock Tickers_ -')

st.markdown('<div style="text-align: justify;">Stock ticker symbols are codes that represent publicly traded companies in the stock market. Although a stocksticker, strictly speaking, is distinct from its ticker symbol -- the ticker is the constantly updating stream of pertinent information pertaining to a stock and the ticker symbol is the three- or four-letter code -- most investors use the term stock ticker as shorthand for its ticker symbol.</div>', unsafe_allow_html=True)
st.write("#")
# 
st.image("https://github.com/J1GS4W2487/ChartW/blob/master/Images/Prof%20Amrita%20Mandalbera.jpeg",caption='Tickers', width=500, use_column_width=00, clamp=False, channels="RGB", output_format="auto")
st.write("#")

st.subheader('_Candlesticks_ -')
st.markdown('<div style="text-align: justify;">Japanese rice traders are responsible for the creation of candlestick patterns. In his 1991 book Japanese Candlestick Charting Techniques, Steve Nison shared these patterns with the West for the first time. Since then, the stock market has been analysed using these candlestick patterns to visualise share price fluctuations. An analyst can forecast changes in share price using these trends. Data from a chosen time frame is combined into one candle bar in candlestick charts. Each candlestick represents activity over a specific period, such as a minute, hour, day, or month. They provide information about price changes for an investment. As a result, a candlestick will display the open, close, high, and low data points for the selected time frame. The colour codes make it clear about changes in the price. Bullish (positive) candlesticks are denoted by green or white, whereas negative bearish(negative) candlesticks are denoted by the colours red or black. occur many times over some time, chart patterns can generally be trusted.</div>', unsafe_allow_html=True)
st.write("#")
st.image('Images\can99.png',caption='Candlesticks', width=400, use_column_width=400, clamp=False, channels="RGB", output_format="auto")

st.write("#")

st.subheader("_Triangle Pattern_ -")
st.markdown('<div style="text-align: justify;">Triangles are most appropriately characterized as horizontal trading patterns. The triangle has maximum width in the beginning. In its most basic form, the triangle depicts waning interest in stocks from both the buy-side and the sell-side: the supply line narrows to satisfy the demand. The trading range narrows and the triangle point is established as the market keeps moving in a sideways pattern. A trend line is a line created over or under pivot highs or lows to represent the current price direction. Such sequences can help investors to understand what the future movement of the stock could be. There are numerous advantages to adopting triangle patterns since they not only signal if a trend will gain momentum or reverse but can also reveal how far the trend continuance or reversal will reach, which can help you evaluate your return-to-risk ratio for the trade. Furthermore, understanding a specific profit target depending on the formed triangles length might help develop and refine your money management approach.</div>', unsafe_allow_html=True)
st.write("#")
st.image('Images\99899.png',caption='Triangle Pattern', width=500, use_column_width=500, clamp=False, channels="RGB", output_format="auto")
st.write("#")

st.subheader('_Simple Moving Average (EMA)_ -')

st.markdown('<div style="text-align: justify;">A statistic known as a moving average measures the typical change over time in a data collection. Moving averages are frequently employed in finance by technical analysts to monitor price patterns for certain stocks. While a negative trend would be interpreted as a sign of deterioration, an upward trend in a moving average could indicate an increase in the price or momentum of an asset. Moving averages come in a wide range of types nowadays, from straightforward measurements to intricate formulas that must be efficiently calculated by a computer program. Technical analysis is a discipline of investing that aims to comprehend and capitalize on the price movement patterns of securities and indices.</div>', unsafe_allow_html=True)
st.write("#")
st.image('Images\SMA.webp',caption='Simple Moving Average', width=500, use_column_width=500, clamp=False, channels="RGB", output_format="auto")
st.write("#")

st.subheader('_Exponential Moving Average (EMA)_ -')

st.markdown("An exponential moving average (EMA) is a sort of moving average (MA) that gives the most recent data points more weight and relevance. The exponentially weighted moving average is another name for the exponential moving average. An exponentially weighted moving average reacts more significantly to recent price changes than a simple moving average (SMA), which applies an equal weight to all observations in the period.")
st.write("#")
st.image('Images\EMA99.JPG',caption='Exponential Moving Average', width=500, use_column_width=500, clamp=False, channels="RGB", output_format="auto")
st.write("#")

st.subheader("Indian Flag Strategy with EMA -")
st.markdown('<div style="text-align: justify;">If you see the EMA lines in the order of the colours present in the Indian Flag, i.e, ORANGE, BLUE and GREEN, it indicates that an upward trend can be followed. In contrast, if the the colours are in a reverse order, i.e, GREEN, BLUE and ORANGE, it forecasts that a downtrend can be traced by the prices.  </div>', unsafe_allow_html=True)
st.write("#")
st.image('Images\EMA.png',caption='Exponential Moving Average', width=500, use_column_width=500, clamp=False, channels="RGB", output_format="auto")
st.write("#")


st.subheader('_Moving Average Convergence Divergence (MACD)_ -')
st.markdown("Moving average convergence/divergence (MACD, or MAC-D) is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a security’s price. The MACD line is calculated by subtracting the 26-period EMA from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of the MACD line is called the signal line, which is then plotted on top of the MACD line, which can function as a trigger for buy or sell signals. Traders may buy the security when the MACD line crosses above the signal line and sell—or short—the security when the MACD line crosses below the signal line. MACD indicators can be interpreted in several ways, but the more common methods are crossovers, divergences, and rapid rises/falls.")
st.write("#")
st.image('Images\MACD.JPG',caption='Moving Average Convergence Divergence', width=500, use_column_width=500, clamp=False, channels="RGB", output_format="auto")
st.write("#")


st.subheader("_Bollinger Bands (BB)_ -")
st.markdown("John Bollinger created the sort of price envelope known as Bollinger Bands. Upper and lower price range levels are indicated by price envelopes. Bollinger Bands are envelops that are drawn above and below a price's simple moving average at a certain standard deviation level. The bands' width adjusts to changes in the underlying price's volatility because it is based on standard deviation.")
st.write("#")
st.image('Images\BB.JPG',caption='Bollinger Bands', width=500, use_column_width=500, clamp=False, channels="RGB", output_format="auto")
st.write("#")

st.subheader('_Relative Strength Index (RSI)_ -')
st.markdown("The relative strength index is a momentum indicator used in technical analysis (RSI). The RSI, which measures the speed and magnitude of recent price fluctuations, is used to evaluate overvalued or undervalued situations in a security's price. The RSI is represented as an oscillator on a scale of 0 to 100. (a line graph). J. Welles Wilder Jr. revealed the indicator he created in his seminal 1978 book New Concepts in Technical Trading Systems. The RSI is useful for more than just identifying overbought and oversold securities. It could also indicate that equities are about to experience a trend reversal or a price correction. It can be used as a buy/sell signal. Historically, an RSI rating of 70 or higher indicates a problem.")
st.write("#")
st.write("#")

