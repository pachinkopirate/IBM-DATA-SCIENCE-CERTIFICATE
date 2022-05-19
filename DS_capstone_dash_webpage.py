# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                dcc.Dropdown(id='site-dropdown',
                                            options=[
                                             {'label': 'All Sites', 'value': 'All Sites'},
                                             {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                             {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                                             {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                             {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                               ],
                                            value='ALL',
                                            placeholder="Select a Launch Site here",
                                            searchable=True
                                            ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                #dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(id='payload-slider',
                                                min=0,
                                                max=10000,
                                                step=1000,
                                                 marks = {
                                                        0: '0 kg',
                                                        1000: '1000 kg',
                                                        2000: '2000 kg',
                                                        3000: '3000 kg',
                                                        4000: '4000 kg',
                                                        5000: '5000 kg',
                                                        6000: '6000 kg',
                                                        7000: '7000 kg',
                                                        8000: '8000 kg',
                                                        9000: '9000 kg',
                                                        10000: '10000 kg'
                                    },
                                                value=[min_payload, max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(site_dropdown):
    filtered_df = spacex_df
    if site_dropdown == 'All Sites':
        fig = px.pie(data_frame=filtered_df,
                    values='class',
                    names='Launch Site',
                    title='Total success for ALL sites')
        return fig
    else:
        site_df=spacex_df.loc[spacex_df['Launch Site'] == site_dropdown]
        fig = px.pie(data_frame=site_df,
                    #values='class',
                    names='class',
                     title='Total Success for site')
        return fig
        # return the outcomes piechart for a selected site
# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              [Input(component_id='site-dropdown', component_property='value'),
              Input(component_id='payload-slider', component_property='value')])
def get_scatter_chart(site_dropdown, payload_slider):
    #filtered_df = spacex_df
    if site_dropdown == 'All Sites':
        low, high = payload_slider
        df  = spacex_df
        mask = (df['Payload Mass (kg)'] > low) & (df['Payload Mass (kg)'] < high)
        fig = px.scatter(data_frame=df[mask],
                         x= 'Payload Mass (kg)',
                         y= 'class',
                         # names='class',
                         #size='Payload Mass (kg)',
                         color = 'Booster Version Category',
                         title='Correlation between Payload and Success')
        print ('in all func')
        return fig
    else:
        print ('in pass')
        low, high = payload_slider
        df  = spacex_df.loc[spacex_df['Launch Site'] == entered_site]
        mask = (df['Payload Mass (kg)'] > low) & (df['Payload Mass (kg)'] < high)
        fig = px.scatter(
            df[mask], x="Payload Mass (kg)", y="class",
            color="Booster Version",
            size='Payload Mass (kg)',
            hover_data=['Payload Mass (kg)'])
        # site_df=spacex_df.loc[spacex_df['Launch Site'] == entered_site]
        # fig = px.pie(data_frame=site_df,
        #             #values='class',
        #             names='class',
        #              title='Total Success for site')
        return fig
        #pass

# Run the app
if __name__ == '__main__':
    app.run_server()