import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import warnings
import matplotlib.dates as mpdates
from plotly.offline import iplot
import cufflinks as cf
import pandas as pd
from datetime import datetime
import pandas_datareader as data
import streamlit as st
import plotly.graph_objects as go
# Import necessary libraries

import pandas as pd
import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp

from dateutil.relativedelta import relativedelta

import datetime
 
Current_Date_Formatted = datetime.datetime.today().strftime ('%Y-%m-%d') # format the date to ddmmyyyy
print ('Current Date: ' + str(Current_Date_Formatted))
 
 
NextDay_Date = datetime.datetime.today() - datetime.timedelta(days=240)
startdate= NextDay_Date.strftime ('%Y-%m-%d') # format the date to ddmmyyyy
print ('Start Date ' + str(startdate))

user_inputs = st.text_input('Enter Stock Ticker 🔠', 'AAPL')

st.write("Starting Date is: ",startdate)
st.write("Today's Date: ",Current_Date_Formatted)

df = yf.download(user_inputs,startdate,Current_Date_Formatted)
df['Id'] = np.arange(1, len(df)+1)

if df.empty:
        st.write("PLEASE ENTER A CORRECT TICKER!")

else:
    cf.go_offline()
    st.write("#")




    st.subheader("Indian Flag Strategy ")
    qf1= cf.QuantFig(df, kind='ohlc', name=user_inputs, title="Flags")
    qf1.add_ema(periods=20,color='Orange')
    qf1.add_ema(periods=50,color='Blue')
    qf1.add_ema(periods=100,color='Green')
    qf1.add_volume()
    fig1=qf1.iplot(asFigure=True, yTitle="Price", rangeslider=True,kind="ohlc")
    st.plotly_chart(fig1)

    st.write("#")
    st.subheader("Bollinger Bands with RSI")
    qf = cf.QuantFig(df, kind='ohlc', name=user_inputs)
    qf.add_rsi(periods=14, color='pink')
    qf.add_bollinger_bands(periods=20, boll_std=2 ,colors=['purple','lime'], fill=True)
    fig=qf.iplot(asFigure=True, yTitle="Price", rangeslider=True,kind="ohlc")
    st.plotly_chart(fig)
        

    
    st.write("#")
    st.subheader("MACD")
    qf2= cf.QuantFig(df, kind='ohlc', name=user_inputs)
    qf2.add_macd(fast_period=15, slow_period=30, signal_period=9, column=None, name='')
    fig3=qf2.iplot(asFigure=True, yTitle="Price", rangeslider=True,kind="ohlc")
    st.plotly_chart(fig3)
















