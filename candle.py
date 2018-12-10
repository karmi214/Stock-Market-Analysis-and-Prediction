
                    graphs = []
                    df=df=get_data(input_data)
                    candlestick = {
                                    'x': df['Date'],
						            'open': df['Open'],
						            'high': df['High'],
						            'low': df['Low'],
						            'close': df['Close'],
						            'type': 'candlestick',
						            'name': ticker,
						            'legendgroup': ticker,
						            'increasing': {'line': {'color': colorscale[0]}},
						            'decreasing': {'line': {'color': colorscale[5]}}
                                    }
                    bb_bands = bbands(df.Close)
                    bollinger_traces = [{
                                         'x': df['Date'], 'y': y,
								         'type': 'scatter', 'mode': 'lines',
								         'line': {'width': 1, 'color': colorscale[(i*2) % len(colorscale)]},
								         'hoverinfo': 'none',
								         'legendgroup': ticker,
								         'showlegend': True if i == 0 else False,
								         'name': '{} - bollinger bands'.format(ticker)                
                                        }for i, y in enumerate(bb_bands)]
                    graphs.append(dcc.Graph(
                     id=ticker,
                     figure={
                     'data': [candlestick] + bollinger_traces,
                     'layout': {
                     'margin': {'b': 0, 'r': 10, 'l': 60, 't': 0},
                     'legend': {'x': 0}
                      }
                     }
                    ))
                    return graphs