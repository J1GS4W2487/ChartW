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
import numpy as np
from matplotlib import pyplot
from scipy.stats import linregress
import pandas as pd

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


        backcandles = 50

        candleid = int(df.Id.max()) - 1

        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

        for i in range(candleid-backcandles, candleid+1):
            if df.iloc[i].Pivot == 1:
                minim = np.append(minim, df.iloc[i].Low)
                xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
            if df.iloc[i].Pivot == 2:
                maxim = np.append(maxim, df.iloc[i].High)
                xxmax = np.append(xxmax, i) # df.iloc[i].name

        #slmin, intercmin = np.polyfit(xxmin, minim,1) #numpy
        #slmax, intercmax = np.polyfit(xxmax, maxim,1)

        if len(xxmin) >= 3 and len(xxmax) >= 3:


            slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
            slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)

            print("Slope of minimum points line:", slmin)
            print("Slope of maximum points line:", slmax)
            print("Difference between the slopes:",(slmax-slmin))

            if (slmax-slmin) < 0.015 and (slmax-slmin)>-0.095 and (slmax-slmin)>-0.028 and abs(rmin)>0.75 :

                print("Channel pattern detected.")

                # Find the intersection point of the two lines
                xi = (intercmin - intercmax) / (slmax - slmin)
                yi = slmin * xi + intercmin

                # Plot the graph with the trendlines and intersection point
                xx = np.concatenate([xxmin, xxmax])
                yy = np.concatenate([minim, maxim])
                fitmin = slmin*xxmin + intercmin
                fitmax = slmax*xxmax + intercmax
        #         df = pd.concat([df, df1])

                fig = go.Figure(data=[go.Candlestick(x=df.Id,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'])])
                fig.add_scatter(x=df.Id, y=df['PointPosition'], mode="markers",
                                marker=dict(size=4, color="MediumPurple"),
                                name="pivot")
                fig.add_trace(go.Scatter(x=xxmin, y=fitmin, mode='lines', name='min slope'))
                fig.add_trace(go.Scatter(x=xxmax, y=fitmax, mode='lines', name='max slope'))

                fig.update_xaxes()
                fig.show()
                print(len(xxmin),xxmax)

            elif (slmin>slmax) and (slmax-slmin)<-4 and abs(rmin)>0.75:
                print("TRI pattern detected.")

                # Find the intersection point of the two lines
                xi = (intercmin - intercmax) / (slmax - slmin)
                yi = slmin * xi + intercmin

                # Plot the graph with the trendlines and intersection point
                xx = np.concatenate([xxmin, xxmax])
                yy = np.concatenate([minim, maxim])
                fitmin = slmin*xxmin + intercmin
                fitmax = slmax*xxmax + intercmax
        #         df = pd.concat([df, df1])

                fig = go.Figure(data=[go.Candlestick(x=df.Id,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'])])
                fig.add_scatter(x=df.Id, y=df['PointPosition'], mode="markers",
                                marker=dict(size=4, color="MediumPurple"),
                                name="pivot")
                fig.add_trace(go.Scatter(x=xxmin, y=fitmin, mode='lines', name='min slope'))
                fig.add_trace(go.Scatter(x=xxmax, y=fitmax, mode='lines', name='max slope'))
                fig.add_trace(go.Scatter(x=[xi], y=[yi], mode='markers', name='intersection',
                                      marker=dict(size=10, color="blue")))

                fig.update_xaxes()
                fig.show()
            elif (slmax-slmin)<-0.03 and (slmax-slmin)>-3.70 and abs(rmin)>0.75:
                print("W pattern detected.")

                # Find the intersection point of the two lines
                xi = (intercmin - intercmax) / (slmax - slmin)
                yi = slmin * xi + intercmin

                # Plot the graph with the trendlines and intersection point
                xx = np.concatenate([xxmin, xxmax])
                yy = np.concatenate([minim, maxim])
                fitmin = slmin*xxmin + intercmin
                fitmax = slmax*xxmax + intercmax
                df = pd.concat([df, df1])

                fig = go.Figure(data=[go.Candlestick(x=df.Id,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'])])
                fig.add_scatter(x=df.Id, y=df['PointPosition'], mode="markers",
                                marker=dict(size=4, color="MediumPurple"),
                                name="pivot")
                fig.add_trace(go.Scatter(x=xxmin, y=fitmin, mode='lines', name='min slope'))
                fig.add_trace(go.Scatter(x=xxmax, y=fitmax, mode='lines', name='max slope'))

                fig.update_xaxes()
                fig.show()


            else:
                  # Find the intersection point of the two lines
                print(" KUCH nahi")
                # Find the intersection point of the two lines
                xi = (intercmin - intercmax) / (slmax - slmin)
                yi = slmin * xi + intercmin

                # Plot the graph with the trendlines and intersection point
                xx = np.concatenate([xxmin, xxmax])
                yy = np.concatenate([minim, maxim])
                fitmin = slmin*xxmin + intercmin
                fitmax = slmax*xxmax + intercmax
                df = pd.concat([df, df1])

                fig = go.Figure(data=[go.Candlestick(x=df.Id,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'])])
                fig.add_scatter(x=df.Id, y=df['PointPosition'], mode="markers",
                                marker=dict(size=4, color="MediumPurple"),
                                name="pivot")
                fig.add_trace(go.Scatter(x=xxmin, y=fitmin, mode='lines', name='min slope'))
                fig.add_trace(go.Scatter(x=xxmax, y=fitmax, mode='lines', name='max slope'))

                fig.update_xaxes()
                fig.show()
                #ulta patterns sab bahar hojaayga
                #bas thoda channel and wedge slopes adjust karna hain


        else:
            print("lesser pivot points")



            
            st.plotly_chart(fig)

        except:
            st.write("No pattern detected")
        if __name__ == '__main__':
            st.set_option('deprecation.showPyplotGlobalUse', False)

