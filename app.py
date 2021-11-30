import base64
import io
from datetime import timedelta
import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table


import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose 


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

def style_row_by_top_values(df, n_bins=9):
    import colorlover
    numeric_columns = df.select_dtypes('number').columns
    styles = []
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    for i in range(df.shape[0]):   # for each row
        row = df.loc[i, numeric_columns].sort_values(ascending=True)
        df_max = row[-1]
        df_min = row[1]
        ranges = [((df_max - df_min) * i) + df_min for i in bounds]

        for j in range(1, len(bounds)):
            min_bound = ranges[j - 1]
            max_bound = ranges[j]
            backgroundColor = colorlover.scales[str(n_bins)]['seq']['Greens'][j - 1]
            color = 'white' if j > len(bounds) / 2. else 'inherit'

            for r in range(len(row)):
                if row[r] > min_bound and row[r] <= max_bound:
                    styles.append({
                        'if': {
                            'filter_query': '{{index}} = "{}"'.format(df['index'][i]),
                            'column_id': row.keys()[r]
                        },
                        'backgroundColor': backgroundColor,
                        'color': color
                    })
    return styles

app.layout = html.Div([ # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
    html.Div([dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drop Here or ',
            html.A('Select File')
        ]),
        style={
            'width': '40%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )], style={'text-align': '-webkit-center'}),
    
    html.Div(id='output')
    
])

# Create content for keyword's groups 
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    groups = pd.read_excel(io.BytesIO(decoded), sheet_name="groups_kw")
    df_groups_ = pd.read_excel(io.BytesIO(decoded), sheet_name="pred_grouped")
    df_ = pd.read_excel(io.BytesIO(decoded), sheet_name="pred_single")

    dict_groups = dict(zip(groups['0'], groups['1']))
    all_groups = set()
    for row in groups['1']:
        all_groups.update({item for item in eval(row)})
    dict_groups['All keyword groups'] = list(all_groups)  

    df_groups = df_groups_.set_index('date').T.reset_index().round(1)
    df_groups_to_decompose = df_groups_.iloc[:-6, :]   
    df_groups_to_decompose['date'] = pd.to_datetime(df_groups_to_decompose['date'])
    data_grouped_t = df_groups_to_decompose.set_index("date")
    seasonal_groups = pd.DataFrame([])
    trend_groups = pd.DataFrame([])
    for query in data_grouped_t.columns:
        a = seasonal_decompose(data_grouped_t[query], model = "add")
        seasonal_groups[query] = a.seasonal.values
        trend_groups[query] = a.trend.values
    min_date = pd.to_datetime(df_groups_['date']).min()
    max_date = pd.to_datetime(df_groups_['date']).max()
    
    
    df = df_.set_index('date').T.reset_index().round(1)
    df_to_decompose = df_.iloc[:-6, :]    
    df_to_decompose['date'] = pd.to_datetime(df_to_decompose['date'])
    data_t = df_to_decompose.set_index("date")
    seasonal = pd.DataFrame([])
    trend = pd.DataFrame([])
    for query in data_t.columns:
        a = seasonal_decompose(data_t[query], model = "add")
        seasonal[query] = a.seasonal.values
        trend[query] = a.trend.values

    kws = [kw for kw in df_.columns[1:]]
    kws.append('All keywords')

    return html.Div([
        html.Hr(),
        html.Div([
        html.Div([
            dcc.Dropdown(
                id = 'drop',
                clearable=False, 
                searchable=False, 
                options=[{'label': i, 'value': i} for i in dict_groups.keys()],
                value='All keyword groups', 
                style= {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#ebb36a'}
            )],style={'width':'30%'}),
            dcc.Store(id="groups_dict", data=dict_groups),
            html.Div([], style={'width':'5%'}),
            html.Div([
                dcc.DatePickerRange(
                    id='my-date-picker-range',  # ID to be used for callback
                    calendar_orientation='horizontal',  # vertical or horizontal
                    day_size=39,  # size of calendar image. Default is 39
                    end_date_placeholder_text="Return",  # text that appears when no end date chosen
                    with_portal=False,  # if True calendar will open in a full screen overlay portal
                    first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
                    reopen_calendar_on_clear=True,
                    is_RTL=False,  # True or False for direction of calendar
                    clearable=False,  # whether or not the user can clear the dropdown
                    number_of_months_shown=1,  # number of months shown when calendar is open
                    min_date_allowed=min_date,  # minimum date allowed on the DatePickerRange component
                    max_date_allowed=max_date,  # maximum date allowed on the DatePickerRange component
                    initial_visible_month=min_date,  # the month initially presented when the user opens the calendar
                    start_date=df_['date'].iloc[-10],
                    end_date=df_['date'].iloc[-1],
                    display_format='YYYY-M-D',  # how selected dates are displayed in the DatePickerRange component.
                    month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.
                    minimum_nights=2,  # minimum number of days between start and end date
                    style = {'height':'150px'},
                    persistence=True,
                    persisted_props=['start_date'],
                    persistence_type='session',  # session, local, or memory. Default is 'local'
                    updatemode='singledate'  # singledate or bothdates. Determines when callback is triggered
                )]),
        ], style={'display': 'flex', 'justify-content': 'center'}),
        html.Hr(),
        html.H5("Grouped keyword search volume", style={'textAlign': 'center', 'margin-top': '50px'}),
        html.Div([html.Div([
            dash_table.DataTable(
                id='table_group',
                columns=[{"name": i, "id": i} 
                        for i in df_groups.columns],
                data=df_groups.to_dict('records'),          
                fixed_columns={ 'headers': True, 'data': 1 },
                style_table={'minWidth': '100%'},
                style_cell={
                    # all three widths are needed
                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'},
                style_data_conditional=style_row_by_top_values(df_groups)
        )], style={'width':'90%'})], style={'text-align':'-webkit-center'}), 
        dcc.Store(id='stored-data-group', data=df_groups.to_dict('records')),
        dcc.Store(id='stored-seasonal-group', data=seasonal_groups.to_dict('records')),
        dcc.Store(id='stored-trend-group', data=trend_groups.to_dict('records')),

        html.Div([ 
            html.Div([
                dcc.Graph(id='plot_group')
            ], style={'display': 'inline-block', 'width':'55%'}),
            html.Div([], style={'display': 'inline-block', 'width':'2%'}),
            html.Div([
                html.Div([dcc.Graph(id='plot_group_seasonality')]), 
                html.Div([dcc.Graph(id='plot_group_trend')]),
            ], style={'display': 'inline-block', 'width':'43%'}),
        ], style={'text-align':'-webkit-center'}),    
        html.Hr(),

        html.H5("Keyword search volume", style={'textAlign': 'center', 'margin-top': '55px'}),
        html.Div([html.Div([
            dash_table.DataTable(
                id='table_single',
                columns=[{"name": i, "id": i} 
                        for i in df.columns],
                data=df.to_dict('records'),          
                fixed_columns={ 'headers': True, 'data': 1 },
                style_table={'minWidth': '100%'},
                style_cell={
                    # all three widths are needed
                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'},
                style_data_conditional=style_row_by_top_values(df)
        )], style={'width':'90%'})], style={'text-align':'-webkit-center'}), 
        dcc.Store(id='stored-data', data=df.to_dict('records')),
        dcc.Store(id='stored-seasonal', data=seasonal.to_dict('records')),
        dcc.Store(id='stored-trend', data=trend.to_dict('records')),

        html.Div([ 
            html.Div([
                dcc.Graph(id='plot_single')
            ], style={'display': 'inline-block', 'width':'55%'}),
            html.Div([], style={'display': 'inline-block', 'width':'2%'}),
            html.Div([
                html.Div([dcc.Graph(id='plot_seasonality')]), 
                html.Div([dcc.Graph(id='plot_trend')]),
            ], style={'display': 'inline-block', 'width':'43%'}),
        ], style={'text-align':'-webkit-center'}),    
    ])


@app.callback(Output('output', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children




# update the data about keyword groups 
@app.callback(
    [Output('table_group', 'data'), Output('table_group', 'columns'), Output('plot_group', 'figure'), Output('plot_group_seasonality', 'figure'),  Output('plot_group_trend', 'figure')],
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input('drop', 'value'), 
     Input('stored-data-group', 'data'),
     Input('stored-seasonal-group', 'data'),
     Input('stored-trend-group', 'data')]
)

def update_data(start_date, end_date, kw, data, seasonal_, trend_):
    df = pd.DataFrame(data)
    seasonal = pd.DataFrame(seasonal_)
    trend = pd.DataFrame(trend_)
    if kw == 'All keyword groups':
        df1 = df.copy()
    else:
        df1 = df[df['index'] == kw]     
    df2 = df1[df1.columns[:-1]]
    df2.columns = pd.to_datetime(df2.columns)
    df3=df2.truncate(after=end_date, before=start_date, axis=1)
    df3.columns = df3.columns.astype(str)
    df_final = df1[['index']]
    for column in df3.columns:
        df_final[column] = df3[column].values
        
    columns =[{"name": i, "id": i} for i in df_final.columns]
        
    fig = go.Figure()
    
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig.add_trace(go.Scatter(x=df_final.set_index('index').T.reset_index()['index'], y=df_final.set_index('index').T.reset_index()[column],
                        mode='lines',
                        name=column))
        if pd.to_datetime(end_date)>= pd.to_datetime(df2.columns[-6]):
            fig.add_vline(x=pd.to_datetime(df2.columns[-6])- timedelta(days=15), line_width=1, line_dash="dash", line_color="green")
    fig.update_layout(title='Search Volume over time', width=500, height=400, legend = dict(font = dict(size = 10, color = "black"),  orientation="h", yanchor="bottom",
    y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=150, b=0))

    fig2 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig2.add_trace(go.Scatter(x=df.set_index('index').T.reset_index()['index'], y=seasonal[column], mode='lines', name=column))
    fig2.update_layout(title='Seasonal Component', width=400, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))
    fig2.update_yaxes(visible=False, showticklabels=False)
    fig3 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig3.add_trace(go.Scatter(x=df.set_index('index').T.reset_index()['index'], y=trend[column], mode='lines', name=column))
    fig3.update_layout(title='Trend Component', width=400, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))
    fig3.update_yaxes(visible=False, showticklabels=False)
    
    return df_final.to_dict('records'), columns , fig, fig2, fig3

# update the data about keywords 

@app.callback(
    [Output('table_single', 'data'), Output('table_single', 'columns'), Output('plot_single', 'figure'), Output('plot_seasonality', 'figure'),  Output('plot_trend', 'figure')],
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input('drop', 'value'), 
     Input('groups_dict', 'data'),
     Input('stored-data', 'data'),
     Input('stored-seasonal', 'data'),
     Input('stored-trend', 'data')]
)

def update_data(start_date, end_date, group, dict_groups, data, seasonal_, trend_):
    df = pd.DataFrame(data)
    seasonal = pd.DataFrame(seasonal_)
    trend = pd.DataFrame(trend_)
    for key, value in dict_groups.items():
        if group == key:
            if group == 'All keyword groups':
                df1 = df[df['index'].isin(value)]
            else:
                df1 = df[df['index'].isin(eval(value))]
           
    df2 = df1[df1.columns[:-1]]
    df2.columns = pd.to_datetime(df2.columns)
    df3=df2.truncate(after=end_date, before=start_date, axis=1)
    df3.columns = df3.columns.astype(str)
    df_final = df1[['index']]
    for column in df3.columns:
        df_final[column] = df3[column].values
        
    columns =[{"name": i, "id": i} for i in df_final.columns]
        
    fig = go.Figure()
    
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig.add_trace(go.Scatter(x=df_final.set_index('index').T.reset_index()['index'], y=df_final.set_index('index').T.reset_index()[column],
                        mode='lines',
                        name=column))
        if pd.to_datetime(end_date)>= pd.to_datetime(df2.columns[-6]):
            fig.add_vline(x=pd.to_datetime(df2.columns[-6])- timedelta(days=15), line_width=1, line_dash="dash", line_color="green")
    fig.update_layout(title='Search Volume over time', width=550, height=400, legend = dict(font = dict(size = 10, color = "black"),  orientation="h", yanchor="bottom",
    y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=150, b=0))

    fig2 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig2.add_trace(go.Scatter(x=df.set_index('index').T.reset_index()['index'], y=seasonal[column], mode='lines', name=column))
    fig2.update_layout(title='Seasonal Component', width=400, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))
    fig2.update_yaxes(visible=False, showticklabels=False)

    fig3 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig3.add_trace(go.Scatter(x=df.set_index('index').T.reset_index()['index'], y=trend[column], mode='lines', name=column))
    fig3.update_layout(title='Trend Component', width=400, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))
    fig3.update_yaxes(visible=False, showticklabels=False)
    
    return df_final.to_dict('records'), columns , fig, fig2, fig3


if __name__ == '__main__':
    app.run_server(debug=True)
