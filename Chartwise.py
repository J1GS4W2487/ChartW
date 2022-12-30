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

# start = '2010-01-01'
# end = '2019-12-31'


st.set_page_config(
    page_title="Chartwise",
     page_icon="📈",
   
)

st.title("ChartWise 📈")
st.sidebar.success("Select a Filter")
# st.write("#")

start1 = st.date_input(
     "Enter Start Date 󠀠󠀠󠀠🟢",
)

end1 = st.date_input(
     "Enter End Date  🔴",
)

if start1==end1:
    st.write("Please enter different dates")

else:
    

    user_input = st.text_input('Enter Stock Ticker 🔠', 'AAPL')
    df = yf.download(user_input,start1,end1)
    df['Id'] = np.arange(1, len(df)+1)

    st.write("#")
    st.write("#")


    if df.empty:
        st.write("ENTER A CORRECT TICKER!")
    else:
        st.subheader('Data Frame')
        st.write(df)
        st.subheader('Candlestick Chart')





        fig = go.Figure(data = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        st.plotly_chart(fig)

        st.subheader('Pivot Points')

        NUM_BEFORE = 3
        NUM_AFTER = 3

        def pivotId(df, candle, num_before, num_after):
            if candle-num_before < 0 or candle+num_after >= len(df):
                return 0
            
            pivotIdLow=1
            pivotIdHigh=1
            for i in range(candle-num_before, candle+num_after):
                if(df.Low[candle]>df.Low[i]):
                    pivotIdLow=0
                if(df.High[candle]<df.High[i]):
                    pivotIdHigh=0
            if pivotIdLow and pivotIdHigh:
                return 3
            elif pivotIdLow:
                return 1
            elif pivotIdHigh:
                return 2
            else:
                return 0
            
        df['Pivot'] = df.apply(lambda row: pivotId(df, int(row.Id)-1, NUM_BEFORE, NUM_AFTER), axis=1)
        def pointPosition(x):
            if x['Pivot']==1:
                return x['Low']-(0.01*df.High.max())
            elif x['Pivot']==2:
                return x['High']+(0.01*df.High.max())
            else:
                return np.nan

        df['PointPosition'] = df.apply(lambda row: pointPosition(row), axis=1)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from datetime import datetime

        fig1 = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])

        fig1.add_scatter(x=df.index, y=df['PointPosition'], mode="markers",
                        marker=dict(size=5, color="MediumPurple"),
                        name="Pivot"
                    )
        #fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig1)

        st.subheader('Triangle Pattern Detection')
        from scipy.stats import linregress

        # Triangle Pattern
        RECENT_HIGH_PIVOT_POINT = 0
        RECENT_LOW_PIVOT_POINT = 0

        FEASIBLE_HIGH_PIVOT_POINTS = np.array([])
        FEASIBLE_LOW_PIVOT_POINTS = np.array([])
        FEASIBLE_HIGH = np.array([])
        FEASIBLE_LOW = np.array([])

        for i in reversed(range(len(df))):
            if df.Pivot[i] == 2 and RECENT_HIGH_PIVOT_POINT == 0:
                RECENT_HIGH_PIVOT_POINT = df.Id[i]
                FEASIBLE_HIGH = np.append(FEASIBLE_HIGH, df.High[i])
            if df.Pivot[i] == 1 and RECENT_LOW_PIVOT_POINT == 0:
                RECENT_LOW_PIVOT_POINT = df.Id[i]
                FEASIBLE_LOW = np.append(FEASIBLE_LOW, df.Low[i])
            if RECENT_HIGH_PIVOT_POINT != 0 and RECENT_LOW_PIVOT_POINT != 0:
                break

        try:
            FEASIBLE_HIGH_PIVOT_POINTS = np.append(FEASIBLE_HIGH_PIVOT_POINTS, RECENT_HIGH_PIVOT_POINT)
            FEASIBLE_LOW_PIVOT_POINTS = np.append(FEASIBLE_LOW_PIVOT_POINTS, RECENT_LOW_PIVOT_POINT)

            dfHigh=df[df['Pivot']==2]
            dfLow=df[df['Pivot']==1]

            MAX_HIGH_PIVOT_POINT = dfHigh.loc[dfHigh['PointPosition'] == dfHigh.PointPosition.max(), 'Id'].iloc[0]
            MIN_LOW_PIVOT_POINT = dfLow.loc[dfLow['PointPosition'] == dfLow.PointPosition.min(), 'Id'].iloc[0]

            FEASIBLE_HIGH_PIVOT_POINTS = np.append(FEASIBLE_HIGH_PIVOT_POINTS, MAX_HIGH_PIVOT_POINT)
            FEASIBLE_HIGH = np.append(FEASIBLE_HIGH, dfHigh.loc[dfHigh['PointPosition'] == dfHigh.PointPosition.max(), 'High'])
            FEASIBLE_LOW_PIVOT_POINTS = np.append(FEASIBLE_LOW_PIVOT_POINTS, MIN_LOW_PIVOT_POINT)
            FEASIBLE_LOW = np.append(FEASIBLE_LOW, dfLow.loc[dfLow['PointPosition'] == dfLow.PointPosition.min(), 'Low'])

            dfHigh = dfHigh[dfHigh['Id'] > MAX_HIGH_PIVOT_POINT]
            dfLow = dfLow[dfLow['Id'] > MIN_LOW_PIVOT_POINT]

            FEASIBLE_HIGH_PIVOT_POINTS = np.append(FEASIBLE_HIGH_PIVOT_POINTS, dfHigh.loc[dfHigh['PointPosition'] == dfHigh.PointPosition.max(), 'Id'].iloc[0])
            FEASIBLE_HIGH = np.append(FEASIBLE_HIGH, dfHigh.loc[dfHigh['PointPosition'] == dfHigh.PointPosition.max(), 'High'])
            FEASIBLE_LOW_PIVOT_POINTS = np.append(FEASIBLE_LOW_PIVOT_POINTS, dfLow.loc[dfLow['PointPosition'] == dfLow.PointPosition.min(), 'Id'].iloc[0])
            FEASIBLE_LOW = np.append(FEASIBLE_LOW, dfLow.loc[dfLow['PointPosition'] == dfLow.PointPosition.min(), 'Low']) 

            slmin, intercmin, rmin, pmin, semin = linregress(FEASIBLE_LOW_PIVOT_POINTS, FEASIBLE_LOW)
            slmax, intercmax, rmax, pmax, semax = linregress(FEASIBLE_HIGH_PIVOT_POINTS, FEASIBLE_HIGH)

            # To show stock data after END DATE to visualize stock trend after pattern
            df1 = yf.download(user_input, start=end1)
            df1['Id'] = np.arange(df.Id.max(), len(df1)+df.Id.max())

            df = pd.concat([df, df1])
            
            fig2 = go.Figure(data=[go.Candlestick(x=df.Id,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'])])

            fig2.add_scatter(x=df.Id, y=df['PointPosition'], mode="markers",
                            marker=dict(size=5, color="MediumPurple"),
                            name="Pivot")

            fig2.add_trace(go.Scatter(x=FEASIBLE_LOW_PIVOT_POINTS, y=slmin*FEASIBLE_LOW_PIVOT_POINTS + intercmin, mode='lines', line=dict(color="purple"), name='min slope'))
            fig2.add_trace(go.Scatter(x=FEASIBLE_HIGH_PIVOT_POINTS, y=slmax*FEASIBLE_HIGH_PIVOT_POINTS + intercmax, mode='lines', line=dict(color="purple"), name='max slope'))

            fig2.update_xaxes(

            )
            
            st.plotly_chart(fig2)

        except:
            st.write("No pattern detected")
        if __name__ == '__main__':
            st.set_option('deprecation.showPyplotGlobalUse', False)


