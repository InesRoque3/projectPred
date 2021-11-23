from datetime import datetime as dt
from datetime import timedelta
import plotly.express as px
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
from dash import dash_table
import plotly.graph_objs as go
# import ast
from statsmodels.tsa.seasonal import seasonal_decompose          

path = 'https://raw.githubusercontent.com/InesRoque3/projectPred/main/data/'

df_ = pd.read_csv('data/Predictions_singleKw.csv')
df = df_.set_index('date').T.reset_index().round(1)
df_to_decompose = df_.iloc[:-6, :]

df_groups_ = pd.read_csv('data/Predictions_groupedKw.csv')
df_groups = df_groups_.set_index('date').T.reset_index().round(1)
df_groups_to_decompose = df_groups_.iloc[:-6, :]

df_to_decompose['date'] = pd.to_datetime(df_to_decompose['date'])
df_groups_to_decompose['date'] = pd.to_datetime(df_groups_to_decompose['date'])
data_t = df_to_decompose.set_index("date")
data_grouped_t = df_groups_to_decompose.set_index("date")

seasonal = pd.DataFrame([])
trend = pd.DataFrame([])
for query in data_t.columns:
  a = seasonal_decompose(data_t[query], model = "add")
  seasonal[query] = a.seasonal.values
  trend[query] = a.trend.values

seasonal_groups = pd.DataFrame([])
trend_groups = pd.DataFrame([])
for query in data_grouped_t.columns:
  a = seasonal_decompose(data_grouped_t[query], model = "add")
  seasonal_groups[query] = a.seasonal.values
  trend_groups[query] = a.trend.values

min_date = pd.to_datetime(df_['date']).min()
max_date = pd.to_datetime(df_['date']).max()

groups = pd.read_csv('data/kw_groups.csv')

dict_groups = dict(zip(groups['0'], groups['1']))
all_groups = set()
for row in groups['1']:
    all_groups.update({item for item in ast.literal_eval(row)})
dict_groups['All keyword groups'] = list(all_groups)

app = dash.Dash(__name__)


# def discrete_background_color_bins(df, n_bins=8, columns='all'):
#     import colorlover
#     bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
#     if columns == 'all':
#         if 'id' in df:
#             df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
#         else:
#             df_numeric_columns = df.select_dtypes('number')
#     else:
#         df_numeric_columns = df[columns]
#     df_max = df_numeric_columns.max().max()
#     df_min = df_numeric_columns.min().min()
#     ranges = [
#         ((df_max - df_min) * i) + df_min
#         for i in bounds
#     ]
#     styles = []
#     legend = []
#     for i in range(1, len(bounds)):
#         min_bound = ranges[i - 1]
#         max_bound = ranges[i]
#         backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
#         color = 'white' if i > len(bounds) / 2. else 'inherit'

#         for column in df_numeric_columns:
#             styles.append({
#                 'if': {
#                     'filter_query': (
#                         '{{{column}}} >= {min_bound}' +
#                         (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
#                     ).format(column=column, min_bound=min_bound, max_bound=max_bound),
#                     'column_id': column
#                 },
#                 'backgroundColor': backgroundColor,
#                 'color': color
#             })
#         legend.append(
#             html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
#                 html.Div(
#                     style={
#                         'backgroundColor': backgroundColor,
#                         'borderLeft': '1px rgb(50, 50, 50) solid',
#                         'height': '10px'
#                     }
#                 ),
#                 html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
#             ])
#         )

#     return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))



#(styles, legend) = discrete_background_color_bins(df_groups)

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

app.layout = html.Div([
    html.Div([
        html.Div([
        dcc.Dropdown(
            id = 'drop_group',
            clearable=False, 
            searchable=False, 
            # multi = True,
            options=[{'label': i, 'value': i} for i in dict_groups.keys()],
            value='All keyword groups', 
            style= {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#ebb36a'}
        )], style={'width':'30%'}),
        dcc.DatePickerRange(
            id='my-date-picker-range',  # ID to be used for callback
            calendar_orientation='horizontal',  # vertical or horizontal
            day_size=39,  # size of calendar image. Default is 39
            end_date_placeholder_text="Return",  # text that appears when no end date chosen
            with_portal=False,  # if True calendar will open in a full screen overlay portal
            first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
            reopen_calendar_on_clear=True,
            is_RTL=False,  # True or False for direction of calendar
            clearable=True,  # whether or not the user can clear the dropdown
            number_of_months_shown=1,  # number of months shown when calendar is open
            min_date_allowed=min_date,  # minimum date allowed on the DatePickerRange component
            max_date_allowed=max_date,  # maximum date allowed on the DatePickerRange component
            initial_visible_month=min_date,  # the month initially presented when the user opens the calendar
            start_date=df_['date'].iloc[-10],
            end_date=df_['date'].iloc[-1],
            display_format='MMM Do, YYYY',  # how selected dates are displayed in the DatePickerRange component.
            month_format='MMMM, YYYY',  # how calendar headers are displayed when the calendar is opened.
            minimum_nights=2,  # minimum number of days between start and end date
            style = {'height':'150px'},
            persistence=True,
            persisted_props=['start_date'],
            persistence_type='session',  # session, local, or memory. Default is 'local'
            updatemode='singledate'  # singledate or bothdates. Determines when callback is triggered
        )
    ], style ={ 'display': 'flex'}) ,
    
    html.H3("Grouped keyword search volume", style={'textAlign': 'center', 'margin-top': '50px'}),
    html.Div([
        dash_table.DataTable(
            id='table_groups',
            columns=[{"name": i, "id": i} 
                    for i in df_groups.columns],
            data=df_groups.to_dict('records'),  #.set_index('date').T.
    #        style_cell=dict(textAlign='left'),
    #         style_header=dict(backgroundColor="paleturquoise"),
    #         style_data=dict(backgroundColor="lavender")
            
            
            fixed_columns={ 'headers': True, 'data': 1 },
            style_table={'maxWidth': '100%'},
            style_cell={
                # all three widths are needed
                'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'},
            style_data_conditional = style_row_by_top_values(df_groups)
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='plot_group')
            ], style={'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='plot_group_seasonality'), 
                dcc.Graph(id='plot_group_trend'),
            ], style={'display': 'inline-block', 'padding-left': '100px'})
        ], id='div2'),
    ], id='div1'),
    html.H3("Keyword search volume", style={'textAlign': 'center', 'margin-top': '55px'}),
    dash_table.DataTable(
        id='table_single',
        columns=[{"name": i, "id": i} 
                 for i in df.columns],
        data=df.to_dict('records'),  #.set_index('date').T.
#         style_cell=dict(textAlign='left'),
#         style_header=dict(backgroundColor="paleturquoise"),
#         style_data=dict(backgroundColor="lavender")
        
        
        fixed_columns={ 'headers': True, 'data': 1 },
        style_table={'minWidth': '100%'},
        style_cell={
            # all three widths are needed
            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis'},
        style_data_conditional = style_row_by_top_values(df)
    ), 
    html.Div([
            dcc.Graph(id='plot_single')
        ], style={'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='plot_seasonality'), 
            dcc.Graph(id='plot_trend'),
        ], style={'display': 'inline-block', 'padding-left': '100px'}),
])

@app.callback(
    [Output('table_groups', 'data'), Output('table_groups', 'columns'), Output('plot_group', 'figure'), Output('plot_group_seasonality', 'figure'),  Output('plot_group_trend', 'figure')],
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'), 
     Input('drop_group', 'value')]
)
def update_data_g(start_date, end_date, group):

# Keyword groups  
    if 'All keyword groups' in group:
        df1 = df_groups.copy()
    else:
        df1 = df_groups[df_groups['index'] == group]
        # df1 = df_groups[df_groups['index'].isin(group)]
        
# rigth format of dataframe 
    df2 = df1[df1.columns[1:]]
    df2.columns = pd.to_datetime(df2.columns)
    
# Select the columns between the date range selected
    df3=df2.truncate(after=end_date, before=start_date, axis=1)
    
# return to the original format
    df3.columns = df3.columns.astype(str)
    df_final = df1[['index']]
    for column in df3.columns:
        df_final[column] = df3[column].values
        
# columns to display
    columns =[{"name": i, "id": i} for i in df_final.columns]

# line plot 
    fig = go.Figure()
    
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig.add_trace(go.Scatter(x=df_final.set_index('index').T.reset_index()['date'], y=df_final.set_index('index').T.reset_index()[column],
                        mode='lines',
                        name=column))
        if pd.to_datetime(end_date)>= pd.to_datetime(df2.columns[-6]):
            fig.add_vline(x=pd.to_datetime(df2.columns[-6])- timedelta(days=15), line_width=1, line_dash="dash", line_color="green")
    fig.update_layout(title='Search Volume over time', width=600, height=400, legend = dict(font = dict(size = 10, color = "black"),  orientation="h", yanchor="bottom",
    y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=150, b=0))

    fig2 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig2.add_trace(go.Scatter(x=data_t.index, y=seasonal_groups[column], mode='lines', name=column))
    fig2.update_layout(title='Seasonal Component', width=500, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))

    fig3 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig3.add_trace(go.Scatter(x=data_t.index, y=trend_groups[column], mode='lines', name=column))
    fig3.update_layout(title='Trend Component', width=500, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))

    return df_final.to_dict('records'), columns, fig, fig2, fig3


@app.callback(
    [Output('table_single', 'data'), Output('table_single', 'columns'), Output('plot_single', 'figure'), Output('plot_seasonality', 'figure'),  Output('plot_trend', 'figure')],
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'), 
     Input('drop_group', 'value')]
)
def update_data(start_date, end_date, group):
    
    for key, value in dict_groups.items():

    # list_kw = []
    # for key, value in dict_groups.items():
    #     if key in group:
    #         for value_ in value:
    #             if value_ not in list_kw:
    #                 list_kw.append(value_) 
    # df1 = df[df['index'].isin(list_kw)]
        if group == key:
            if group == 'All keyword groups':
                df1 = df[df['index'].isin(value)]
            else:
                df1 = df[df['index'].isin(eval(value))] #ast.literal_eval
            
    df2 = df1[df1.columns[1:]]
    df2.columns = pd.to_datetime(df2.columns)
    df3=df2.truncate(after=end_date, before=start_date, axis=1)
    df3.columns = df3.columns.astype(str)
    df_final = df1[['index']]
    for column in df3.columns:
        df_final[column] = df3[column].values
        
    columns =[{"name": i, "id": i} for i in df_final.columns]
        
    fig = go.Figure()

    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig.add_trace(go.Scatter(x=df_final.set_index('index').T.reset_index()['date'], y=df_final.set_index('index').T.reset_index()[column],
                        mode='lines',
                        name=column))
        fig.add_vline(x=pd.to_datetime(df2.columns[-6])- timedelta(days=15), line_width=1, line_dash="dash", line_color="green")
    fig.update_layout(title='Search Volume over time', width=600, height=400, legend = dict(font = dict(size = 10, color = "black"),  orientation="h", yanchor="bottom",
    y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=250, b=0))

    fig2 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig2.add_trace(go.Scatter(x=data_t.index, y=seasonal[column], mode='lines', name=column))
    fig2.update_layout(title='Seasonal Component', width=500, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))

    fig3 = go.Figure()
    for column in df_final.set_index('index').T.reset_index().columns[1:]:
        fig3.add_trace(go.Scatter(x=data_t.index, y=trend[column], mode='lines', name=column))
    fig3.update_layout(title='Trend Component', width=500, height=175, legend = dict(font = dict(size = 10, color = "black")), margin=dict(l=0, r=0, t=50, b=0))

    
    return df_final.to_dict('records'), columns, fig, fig2, fig3


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)