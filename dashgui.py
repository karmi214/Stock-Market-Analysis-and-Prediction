import dash
from dash.dependencies import Output, Event, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import colorlover as cl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import style
import random as r
yf.pdr_override()

stock_data=[]
cur_stock=''
predicted_stock=[]
def get_data(input_data):
	global cur_stock
	global stock_data
	#stock_data=pd.read_csv('TSLA.csv')
	#make_prediction()
	#return stock_data
	if cur_stock!=input_data:
		start = datetime.datetime(2000, 1, 1)
		end = datetime.datetime.now()		    
		df = pdr.get_data_yahoo(input_data, start=start, end=end)
		stock_data=df
		cur_stock=input_data
		make_prediction()
		#print(df)
	return stock_data


colorscale = cl.scales['9']['qual']['Paired']

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def next_prediction():
	global stock_data
	global scalar
	global predicted_stock
	#global regressor
	try:
		df=stock_data
		new=np.zeros_like(df)
		dataset = df.iloc[:, [1,4]].values
		predict=np.array(dataset)
		for i in range(0,2):
			new=new.dot(predict)
			new.append()
	except:
		pass
	return predict


from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler(feature_range=(0, 1))


import keras
from keras.models import Sequential
from keras.models import load_model
def make_prediction():
	global stock_data
	global scalar
	global predicted_stock
	df=stock_data
	dataset = df.iloc[:, [1,4]].values
	X = df.iloc[:, 1].values
	y = df.iloc[:, 4].values
	dataset_scaled = scaler.fit_transform(dataset)
	X = dataset_scaled[:, 0]
	y = dataset_scaled[:, 1]
	#print(dataset_scaled)
	all_real_stock_price = np.array(y)
	inputs = np.array(X)
	regressor = load_model('model.h5')
	all_predicted_stock_price = regressor.predict(inputs)
	# rebuild the Structure
	dataset_test_total = pd.DataFrame()
	dataset_test_total['real'] = all_real_stock_price
	dataset_test_total['predicted'] = all_predicted_stock_price
	# real test data price VS. predicted price
	predicted_stock_price = scaler.inverse_transform(dataset_test_total)
	predicted_stock=predicted_stock_price[:,1]
	print(predicted_stock)
	



df_symbol = pd.read_csv('tickers.csv')
app = dash.Dash()

colors = {
    'background': '#C9543B',
    'text': '#428BCA'
}

app.title='SM-A & P'
app.layout = html.Div([
    html.Img(src="https://st3.depositphotos.com/2485091/18274/v/450/depositphotos_182748326-stock-illustration-stock-market-concept-design-with.jpg",
                style={
                    'height': '220px',
                    'width': '230px',
                     'display': 'block',
   					 'margin-left': 'auto',
    				 'margin-right': 'auto',
                },
        ),
    html.Div([
        html.H1('Stock Market Analysis and Prediction',
                style={
                       'textAlign': 'center',
                       'color': colors['text'],
                       'font-size': '2.65em',
                       'font-family': 'Product Sans',
                       }),
        
    ]),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': s[0], 'value': str(s[1])}
                 for s in zip(df_symbol.Company, df_symbol.Symbol)],
        #value=['AAPL'],
        #multi=True
    ),
    html.Div([
    	dcc.RadioItems(
                id='graph_plot',
                options=[{'label': i, 'value': i} for i in ['All data','Previous 30','Previous 7', 'Future prediction','Candle-stick for 30 days']],
                value='All data',
                labelStyle={'display': 'inline-block','margin':'15px',},
                style={'textAlign':'center',
                'padding':'10px',
                'font-family':'"Arial Narrow", Arial, "Helvetica Condensed", Helvetica, sans-serif',
                'color':'green',
                'font-size':'20px',
                },

            ),
    	#html.Button('7 days prediction', id='predict-btn',style={'display':'inline','background-color': '#FFF','margin-top':'20px','border-radius': '6px','border': '2px solid #4CAF50'}),
    	#html.Button('Candle', id='candle-btn',style={'display':'inline','background-color': '#FFF','margin-left':'10px','margin-top':'20px','border-radius': '6px','border': '2px solid #4CAF50'}),
    ]),
    html.Div([
    	html.Div(children='Show prediction: ',style={'display':'inline-block','float':'left','margin':'5px'} ),
    	dcc.RadioItems(
                id='prediction_btn',
                options=[{'label': i, 'value': i} for i in ['Yes','No']],
                value='No',
                labelStyle={'display': 'inline-block','margin':'5px'}
            ),]),
    html.Div(id='predict-graph'),
    html.Div(id='output-graph',style={'border':'1px thin white',
        'padding': '20px',
        'margin':'50px 5px 20px 5px',
        'box-shadow': '2px 8px 5px 8px #888888',
       })
], className="container",style={'padding':'0px 10px 60px 10px'})


@app.callback(
    dash.dependencies.Output('output-graph', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value'),dash.dependencies.Input('graph_plot', 'value'),dash.dependencies.Input('prediction_btn', 'value')])
def update_value(input_data,graph_plot,prediction_btn):
	global predicted_stock
	if graph_plot=='All data':
		try:
		   	df=get_data(input_data)
		   	print(df)
		   	ds=[]
		   	#print('now')
		   	if prediction_btn=='Yes':
		   		#print('prediction dekhuane')
		   		#global predicted_stock
		   		ds=predicted_stock
		   	return html.Div([dcc.Graph(
		        id='example-graph',
		        figure={
		            'data': [
		                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': 'Real closing'},
		                {'x': df.index, 'y': ds, 'type': 'line', 'name': 'Predicted closing'},
		            ],
		            'layout': {
		                'title': input_data,
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Closing Price(in USD)',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))
		            }
		        }
		    ),dcc.Graph(
		    	id='bar',
		    	figure={
		    		'data':[
		    			{'x':df.index,'y':df.Volume, 'type':'bar', 'name': 'Volume'},
		    		],
		    		'layout':{'title': 'Volume',
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Volume',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))}
		    	}

		    	)
		    ])
		except:
			return html.H1(children='No data available',
				style={
                       'textAlign': 'center',
                       'color': colors['background'],
                       'font-size': '1.45em',
                       'font-family': 'Product Sans',
                       'padding':'10',
                      })
	elif graph_plot=='Previous 7':
		try:
		    df = get_data(input_data)
		    print(df)
		    ds=[]
		    if prediction_btn=='Yes':
		   		#print('prediction dekhuane')
		   		ds=predicted_stock
		   		ds=ds[-7:]
		   		#print(ds)
		    df=df.tail(7)
		    return html.Div([dcc.Graph(
		        id='example-graph',
		        figure={
		            'data': [
		                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': 'Real closing'},
		                {'x': df.index, 'y': ds, 'type': 'line', 'name': 'Predicted closing'},
		            ],
		            'layout': {
		                'title': input_data,
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Closing Price(in USD)',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))
		            }
		        }
		    ),
		    dcc.Graph(
		    	id='bar',
		    	figure={
		    		'data':[
		    			{'x':df.index,'y':df.Volume, 'type':'bar', 'name': 'Volume'},
		    		],
		    		'layout':{'title': 'Volume',
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Volume',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))}
		    	}

		    	)]
		    )
		except:
			return html.H1(children='No data available',
				style={
                       'textAlign': 'center',
                       'color': colors['background'],
                       'font-size': '1.45em',
                       'font-family': 'Product Sans',
                       'padding':'10',
                       })
	elif graph_plot=='Previous 30':
		try:
		    df = get_data(input_data)
		    print(df)
		    df=df.tail(30)
		    ds=[]
		    if prediction_btn=='Yes':
		   		#print('prediction dekhuane')
		   		ds=predicted_stock
		   		ds=ds[-30:]
		    return html.Div([dcc.Graph(
		        id='example-graph',
		        figure={
		            'data': [
		                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': 'Real closing'},
		                {'x': df.index, 'y': ds, 'type': 'line', 'name': 'Predicted closing'},
		            ],
		            'layout': {
		                'title': input_data,
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Closing Price(in USD)',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))
		            }
		        }
		    ),dcc.Graph(
		    	id='bar',
		    	figure={
		    		'data':[
		    			{'x':df.index,'y':df.Volume, 'type':'bar', 'name': 'Volume'},
		    		],
		    		'layout':{'title': 'Volume',
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Volume',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))}
		    	}

		    	)
		    ])
		except:
			return html.H1(children='No data available',
				style={
                       'textAlign': 'center',
                       'color': colors['background'],
                       'font-size': '1.45em',
                       'font-family': 'Product Sans',
                       'padding':'10',
                       })
	elif graph_plot=='Future prediction':
		try:
			df=get_data(input_data)
			print(df)
			df=df.tail(4)
			last=df.tail(3)
			last=last['Close'].values
			#last=last[2]
			#print(last)
			sd=last[1]-last[0]
			#print(last)
			m=last[2]-last[1]
			last=last[2]
			date=[0,1,2]
			close=[0,1,2]
			#last=df.tail(1)
			#last=df.Close
			#future_value=next_prediction()
			#print(last)
			r.seed(sd)
			for i in range(0,3):
				date[i]=datetime.date.today()+datetime.timedelta(days=i-1)
				if i==0:
					close[i]=last
				else:
					close[i]=last+m*r.uniform(-1.0, 1.0)
					last=close[i]
			prediction={'Date':date,'Close':close}
			ds=pd.DataFrame(prediction)
			dss=next_prediction()
			return html.Div([dcc.Graph(
		        id='example-graph',
		        figure={
		            'data': [
		                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': 'Real closing'},
		                {'x': ds.Date, 'y': ds.Close, 'type': 'line', 'name': 'Predicted closing'},
		            ],
		            'layout': {
		                'title': input_data,
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                title='Closing Price(in USD)',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'))
		            }
		        }
		    ),html.Div(html.H5(children='Disclaimer: The predicted value may not match the real one. So trade at your own risk'),
		    	style={ 'color': '#F00',
		    			'font-size': '.75em',
                       	'font-family': 'Product Sans',
                       	'padding':'10',
                       	'margin-left':'30',
		    	})
		    ])
		except:
			return html.H1(children='No data available',
				style={
                       'textAlign': 'center',
                       'color': colors['background'],
                       'font-size': '1.45em',
                       'font-family': 'Product Sans',
                       'padding':'10',
                       })
	else: 
	    for ticker in enumerate(input_data):
	        try:
	            ds = get_data(input_data)
	            #print(ds)
	            df=ds.tail(30)
	            print(ds)
	            rss=rsiFunc(ds.Close)
	            #print(len(rss))
	            rsss={'rsi':rss}
	            rsii=pd.DataFrame(rsss)
	            rsii=rsii.tail(30)
	            u=rss
	            #u.rsi=30
	            #u=u.rsi
	            o=rss
	            for i in range(0,30):
	            	u[i]=30
	            	o[i]=70
	        except:
	            return html.H1(children='No data available',
				style={
                       'textAlign': 'center',
                       'color': colors['background'],
                       'font-size': '1.45em',
                       'font-family': 'Product Sans',
                       'padding':'10',
                       })
	            continue

	        candlestick = {
	            'x': df.index,
	            'open': df['Open'],
	            'high': df['High'],
	            'low': df['Low'],
	            'close': df['Close'],
	            'type': 'candlestick',
	            'name': input_data,
	            'legendgroup': input_data,
	            'Increasing': {'line': {'color': colorscale[3]}},
	            'Decreasing': {'line': {'color': colorscale[5]}}
	        }
	        return html.Div([dcc.Graph(
	            id='example-graph',
	            figure={
	                'data': [candlestick],
	                'layout': {
	                    'margin': {'b': 0, 'r': 10, 'l': 60, 't': 0},
	                    'legend': {'x': 0},
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                size=20,
			                color='#7f7f7f'
			            )),
			            
	                    

	                }
	            }
	        ),html.Div([
	        	html.Img(src='https://i.investopedia.com/inv/dictionary/terms/candle.gif',
	        		style={'float':'left','display':'inline-block','height': '220px','width': '230px','margin-left':'40px'}),

	        	html.Div(children='A candlestick is a chart that displays the high, low, opening and closing prices of a security for a specific period. The wide part of the candlestick is called the "real body" and tells investors whether the closing price was higher or lower than the opening price. Black/red indicates that the stock closed lower and white/green indicates that the stock closed higher.',
	        		style={'font-family': 'Product Sans','font-size':'1.25em','margin':'20px','margin-left':'300px','color': colors['background']}),

	        ],id='details',style={'height':'220px','margin':'20px'}
	        ),dcc.Graph(
	        	id='rsi',
	        	figure={
	        		'data':[
		    			{'x':df.index,'y':rsii.rsi, 'type':'line', 'name': 'RSI'},
		    			{'x':df.index,'y':u-40, 'type':'line', 'name': 'Oversold line'},
		    			{'x':df.index,'y':o, 'type':'line', 'name': 'Overbought line'},
		    		],
		    		'layout':{
		    		'title': 'Relative strength index',
			            'xaxis' : dict(
			                title='Date',
			                titlefont=dict(
			                family='Courier New, monospace',
			                
			                size=20,
			                color='#7f7f7f'
			            )),
			            'yaxis' : dict(
			                range=[0,100],
			            ),
			            }

	        	}
	        ),html.Div(children='The relative strength index (RSI) is a momentum indicator developed by noted technical analyst Welles Wilder, that compares the magnitude of recent gains and losses over a specified time period to measure speed and change of price movements of a security. It is primarily used to attempt to identify overbought or oversold conditions in the trading of an asset.',
	        style={'display':'inline-block','color': colors['background'],'font-size': '1.25em','font-family': 'Product Sans','padding':'10','margin':'50','margin-top':'-10px'}
	        )
	        ])                 
             
          



external_css = ["https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css"]

for css in external_css:
    app.css.append_css({"external_url": css})


if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })


if __name__ == '__main__':
    app.run_server(debug=True)
