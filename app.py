import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dependencies
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from ApiCaller import ApiCaller
from ModelGenerator import ModelGenerator

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

app = dash.Dash()

server = app.server

app.title = 'Footbalytics'

# Obtaining data from the API

apicaller = ApiCaller(2022)
leagues = apicaller.get_leagues()
unique_countries = leagues.dropna(subset=['country_code']).drop_duplicates(subset=['country_name', 'country_code'])
country_options = [{'label': row['country_name'], 'value': row['country_code']} for index, row in unique_countries.iterrows()]

# Obtaining data from ModelGenerator

modelgenerator = ModelGenerator()
regression_numeric_columns = modelgenerator.get_numeric_columns()
regression_columns = modelgenerator.get_all_cols()

# Defining the layout of the app

app.layout = html.Div([
    html.Header([
        html.Div([
            html.Img(src='/assets/logo.png', className='header-logo'),
            dcc.Link(html.H1('Footbalytics', className='header-title', style={'color': 'black', 'cursor': 'pointer', 'transition': '0.2s'}), href='/')
        ], className='header-left'),

        html.Div([
            dcc.Link('Team analytics', href='/dashboard', className='nav-link'),
            dcc.Link('Machine learning models', href='/machine-learning', className='nav-link')
        ], className='header-right')
    ], className='header'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# homepage layour

layout_home = html.Div(id='layout_home', children=[
        # Base Page Layout
        html.Div([
            # Left side: GIF
            html.Div([
                html.Img(src='/assets/mourinho.gif', width="480", height="480", className="giphy-embed"),
            ], style={'display': 'inline-block', 'vertical-align': 'middle'}),  # Align the image to the middle

            # Right side: Title and text
            html.Div([
                html.H1 ("Welcome!", className='title-text'),  # Apply the title-text class
                html.P("Thanks to Footbalytics, you are in control of the data.", className='text-content'),  # Apply the text-content class
                html.P("In this website, you will be able to easily generate dashboards and machine learning models of your favourite football teams.", className='text-content'),  # Apply the text-content class
            ], style={'display': 'inline-block', 'margin-left': '20px', 'vertical-align': 'middle', 'width': '20%'}),  # Align the text to the middle
        ], style = {"margin-top": "70px", "margin-left": "30%"})
    ])

# dashboard layout

layout_dashboard = html.Div(
    children = [
        html.H1( # Título de la página
            children = [
                "Dashboard"
            ],
            id = "dahsboard_title",
            style = {
                "text-align": "Left",
                "font-family": "Arial",
            }
        ),
        html.Div(  # Container for the form
            children=[
                dcc.Dropdown(
                    options=country_options,
                    placeholder="Select a country",
                    id="dropdown_country",
                    style={
                        "display": "inline-block",
                        "width": "175px",  # Adjusted width
                        "margin-right": "30px",  # Adjusted margin
                    }
                ),
                dcc.Dropdown(
                    placeholder="Select a league",
                    id="dropdown_league",
                    style={
                        "display": "inline-block",
                        "width": "250px",  # Adjusted width
                        "margin-right": "30px",  # Adjusted margin
                    }
                ),
                dcc.Dropdown(
                    placeholder="Select a team",
                    id="dropdown_team",
                    style={
                        "display": "inline-block",
                        "width": "250px",  # Adjusted width
                    }
                ),
            ],
            style={
                "font-family": "Arial",
                "display": "flex",  # Use flexbox layout
                "justify-content": "center",  # Center items horizontally
                "align-items": "center",  # Center items vertically
                "height": "100px",  # Adjust height as needed
                "margin-bottom": "10px",  # Margin for the entire form container
            }
        ),
        html.Div(  # Container for the title and image
            children=[
                html.Img(
                    src="your-placeholder-image-url",  # Placeholder for the image source
                    id="dashboard_image",
                    style={
                        "width": "100px",  # Adjust width as needed
                        "height": "100px",  # Adjust height as needed
                        "display": "inline-block",  # Display inline with the title
                        "vertical-align": "middle",  # Align vertically with the title
                        "margin-right": "10px",  # Margin between image and title
                    },
                    alt="Image not found"
                ),
                html.H2(
                    children=[
                        "Team stats dashboard"
                    ],
                    id="dashboard_subtitle",
                    style={
                        "text-align": "left",
                        "display": "inline-block",
                        "vertical-align": "middle",  # Align vertically with the image
                        "font-family": "Arial"
                    }
                )
            ],
            id="dashboard_subtitle_container",
            style={
                "font-family": "Arial",
                "display": "none",  # Use flexbox for alignment
                "align-items": "center",  # Center items vertically
                "justify-content": "center",  # Center items horizontally
            }
        ),
        html.Div(  # Figures of clean sheets and pie chart
            children=[
                html.Div(
                    dcc.Graph(id="figure_clean_sheets",
                              style={"display": "none"}),
                    style={"width": "35%"}
                ),
                html.Div(
                    dcc.Graph(id="figure_wdl",
                              style={"display": "none"}),
                    style={"width": "35%"}
                )
            ],
            style={"text-align": "center",
                   "display" : "inline-block"}
        ),
        html.Div(  # Heatmap and goals
            children=[
                html.Div(
                    dcc.Graph(id="figure_wld_heatmap",
                              style={"display": "none"}),
                    style={"width": "35%"}
                ),
                html.Div(
                    dcc.Graph(id="figure_goals",
                              style={"display": "none"}),
                    style={"width": "35%"}
                )
            ],
            style={"text-align": "center",
                   "display" : "inline-block"}
        ),
        html.Div( # Container for the cards
            dcc.Graph(
                id = "figure_cards",
                style = {
                    "display": "none"
                }
            )  
        )
    ],
    id = "dashboard_page",
    style = {
        "margin-right": "125px",
        "margin-left": "125px",
        "margin-top": "50px",
    } 
)

# machine learning layout

layout_machine_learning = html.Div(
    children=[
        # Section title and introduction
        html.Div(
            children=[
                html.H2("FIFA Player Data Analysis"),
                html.P("Generate a linear regression model using official FIFA player statistics. Explore how different attributes relate to a player's overall rating, value, and more."),
                html.P("At the end of this page, you can see the definitions of the variables used in this analysis. Be careful! If you add a categorical variable with many unique values, the model will generate dummy variables, which are more difficult to understand and make the model more complex!")
            ],
            style={"margin-bottom": "20px"}
        ),
        html.Div(
            children=[
                # Left side: Forms
                html.Div(
                    children=[
                        # Top form (small)
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    options=regression_numeric_columns,
                                    placeholder="Select a target variable",
                                    id="dropdown_regression_target",
                                    style={"width": "250px", "margin-top": "80px"}
                                )
                            ],
                            style={"margin-bottom": "10px"}
                        ),
                        # Large form with scrollbar
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    placeholder="Select the training variables",
                                    id="dropdown_regression_input",
                                    style={"width": "400px", "position": "relative", "zIndex": "999"},
                                    multi=True
                                ),
                                # Button below the large form
                                html.Button("Generate model", id="regression_button", 
                                            style={"width": "400px", "margin-top": "10px", "height": "30px", 
                                                   "backgroundColor": "black", "color": "white", "fontWeight": "bold"}),
                            ],
                            style={"margin-bottom": "10px"}
                        )
                    ],
                    style={"flex": "1"}
                ),
                # Right side: Graph
                html.Div(
                    children=[
                        dcc.Graph(id="regression_matrix", style={"width": "100%", 'display': "none"}) 
                    ],
                    style={"flex": "2"}
                )
            ],
            style={"display": "flex", "flex-direction": "row"}
        ),
        # Bottom section: Model result
        html.Div(
            children=[],
            id="regression_result",
            style={'display': "block"}
        ),
        # Variable explanations
        html.Div(
            children=[
                html.H3("Variable Descriptions:"),
                html.Ul(children=[
                    html.Li("Name: Name of the player."),
                    html.Li("Age: Age of the player."),
                    html.Li("Height (cm): Player's height in centimeters."),
                    html.Li("Weight (kg): Player's weight in kilograms."),
                    html.Li("Positions: Positions the player can play."),
                    html.Li("Overall Rating: Overall rating of the player in FIFA."),
                    html.Li("Value (Euro): Market value of the player in euros."),
                    html.Li("Wage (Euro): Weekly wage of the player in euros."),
                    html.Li("Preferred Foot: Player's preferred foot."),
                    html.Li("Crossing: Rating for crossing ability."),
                    html.Li("Finishing: Rating for finishing ability."),
                    html.Li("Heading Accuracy: Rating for heading accuracy."),
                    html.Li("Short Passing: Rating for short passing ability."),
                    html.Li("Volleys: Rating for volleys."),
                    html.Li("Dribbling: Rating for dribbling."),
                    html.Li("Curve: Rating for curve shots."),
                    html.Li("Freekick Accuracy: Rating for free kick accuracy."),
                    html.Li("Long Passing: Rating for long passing."),
                    html.Li("Ball Control: Rating for ball control."),
                    html.Li("Acceleration: Rating for acceleration."),
                    html.Li("Sprint Speed: Rating for sprint speed."),
                    html.Li("Agility: Rating for agility."),
                    html.Li("Reactions: Rating for reactions."),
                    html.Li("Balance: Rating for balance."),
                    html.Li("Shot Power: Rating for shot power."),
                    html.Li("Jumping: Rating for jumping."),
                    html.Li("Stamina: Rating for stamina."),
                    html.Li("Strength: Rating for strength."),
                    html.Li("Long Shots: Rating for long shots."),
                    html.Li("Aggression: Rating for aggression."),
                    html.Li("Interceptions: Rating for interceptions."),
                    html.Li("Positioning: Rating for positioning."),
                    html.Li("Vision: Rating for vision."),
                    html.Li("Penalties: Rating for penalties."),
                    html.Li("Composure: Rating for composure."),
                    html.Li("Marking: Rating for marking."),
                    html.Li("Standing Tackle: Rating for standing tackle."),
                    html.Li("Sliding Tackle: Rating for sliding tackle.")
                ]),
            ],
            style={"margin-top": "20px"}
        )
    ],
    id="machine_learning_page",
    style={
        "margin-right": "125px",
        "margin-left": "125px",
        "margin-top": "50px",
        "font-family": "Arial",
        "display": "flex",  # Use flexbox layout
        "flex-direction": "column",  # Align children vertically
    }
)





# navigation between pages

@app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/dashboard':
        return layout_dashboard
    elif pathname == '/machine-learning':
        return layout_machine_learning
    elif pathname == '/':
        return layout_home
    else:
        return 'Page not found'

# callbacks for the dashboard

# form options

@app.callback(
    Output('dropdown_league', 'options'),
    Input('dropdown_country', 'value')
)
def set_league_list(selected_country):

    # Filter the DataFrame based on country_code
    filtered_table = leagues[leagues['country_code'] == selected_country][['league_name', 'league_id']].drop_duplicates()
    # Update the options for the league dropdown
    return [{'label': row_data['league_name'], 'value': row_data['league_id']} for index, row_data in filtered_table.iterrows()]


@app.callback(
    Output('dropdown_team', 'options'),
    Input('dropdown_league', 'value')
)
def set_team_list(selected_league):
    teams = apicaller.get_teams_from_league(selected_league)
    # Update the options for the second dropdown
    return [{'label': row_data['team_name'], 'value': row_data['team_id']} for index, row_data in teams.iterrows()]


# dashboard
@app.callback(
    Output('dashboard_image', 'src'),
    Output('dashboard_subtitle', 'children'),
    Output('dashboard_subtitle_container', 'style'),
    Output('figure_cards', 'figure'),
    Output('figure_cards', 'style'),
    Output('figure_clean_sheets', 'figure'),
    Output('figure_clean_sheets', 'style'),
    Output('figure_wdl', 'figure'),
    Output('figure_wdl', 'style'),
    Output('figure_wld_heatmap', 'figure'),
    Output('figure_wld_heatmap', 'style'),
    Output('figure_goals', 'figure'),
    Output('figure_goals', 'style'),
    Input('dropdown_team', 'value'),
    Input('dropdown_league', 'value')
)
def update_figures(selected_team, selected_league):
    if selected_team is None or selected_league is None:
        return ('your-placeholder-image-url', 'Team stats dashboard', {'display': 'none'}, {}, {"display": "none"}, {}, {"display": "none"}, {}, {"display": "none"}, {}, {"display": "none"}, {}, {"display": "none"})
    team_stats = apicaller.get_team_stats(selected_team, selected_league)

    # logo and title information

    logo = team_stats['team_logo'].iloc[0]
    subtitle = team_stats['team_name'].iloc[0] + ' stats dashboard (22-23 season)'

    # retrieving team stats

    team_stats = apicaller.get_team_stats(selected_team, selected_league)

    
    ## CARDS PLOT
    # Time slots for the cards
    time_slots = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '91-105', '106-120']

    # Extracting yellow and red card data for each time slot

    yellow_cards = []
    red_cards = []

    for slot in time_slots:
        yellow_cards.append(team_stats['cards_yellow_' + slot + '_total'].iloc[0])
        red_cards.append(team_stats['cards_red_' + slot + '_total'].iloc[0])

    # Creating the bar plot
    fig_cards = go.Figure()
    fig_cards.add_trace(go.Bar(x=time_slots, y=yellow_cards, name='Yellow Cards', marker_color='#D9B400'))
    fig_cards.add_trace(go.Bar(x=time_slots, y=red_cards, name='Red Cards', marker_color='#DE3E3E')) 

    # Adding title and labels
    fig_cards.update_layout(
        title='Red and yellow cards',
        xaxis_title='Time slot',
        yaxis_title='Card count',
        barmode='stack',  # Change barmode to stack
        plot_bgcolor='#FFFFFF',
    )

    ## CLEAN SHEETS PLOT
    # getting the data

    fig_clean_sheets_x = ['home', 'away', 'total']
    clean_sheets = []
    failed_to_score = []

    for y in fig_clean_sheets_x:
        clean_sheets.append(team_stats['clean_sheet_' + y ].iloc[0])
        failed_to_score.append(team_stats['failed_to_score_' + y ].iloc[0])
    
    # Creating the bar plot

    fig_clean_sheets = go.Figure()
    fig_clean_sheets.add_trace(go.Bar(x=fig_clean_sheets_x, y=clean_sheets, name='Clean Sheets', marker_color='#D9B400'))
    fig_clean_sheets.add_trace(go.Bar(x=fig_clean_sheets_x, y=failed_to_score, name='Failed to score', marker_color='#FF6E6E'))

    # Adding title and labels
    fig_clean_sheets.update_layout(
        title='Clean sheets vs. no goals',
        xaxis_title='Match outcome',
        yaxis_title='Match count',
        barmode='stack',  # Change barmode to stack
        plot_bgcolor='#FFFFFF'
    )


    ## WDL PLOT
    # getting the data
    wins_home = team_stats['fixtures_wins_home'].iloc[0]
    wins_away = team_stats['fixtures_wins_away'].iloc[0]
    wins_total = team_stats['fixtures_wins_total'].iloc[0]

    draws_home = team_stats['fixtures_draws_home'].iloc[0]
    draws_away = team_stats['fixtures_draws_away'].iloc[0]
    draws_total = team_stats['fixtures_draws_total'].iloc[0]

    loses_home = team_stats['fixtures_loses_home'].iloc[0]
    loses_away = team_stats['fixtures_loses_away'].iloc[0]
    loses_total = team_stats['fixtures_loses_total'].iloc[0]

    total = wins_total + draws_total + loses_total

    # Creating the sunburst chart for match outcomes
    fig_wdl = go.Figure(go.Sunburst(
        labels=["Total", "Wins", "Wins Home", "Wins Away", "Draws", "Draws Home", "Draws Away", "Losses", "Losses Home", "Losses Away"],
        parents=["", "Total", "Wins", "Wins", "Total", "Draws", "Draws", "Total", "Losses", "Losses"],
        values=[total, wins_total, wins_home, wins_away, draws_total, draws_home, draws_away, loses_total, loses_home, loses_away],
        branchvalues="total",
        marker=dict(colors=['#FFFFFF', '#61C750', '#9ACD32', '#556B2F', '#FDFF6E', '#FEFFA8', '#CECF78', '#DE3E3E', '#DC143C', '#B22222'])
    ))

    # Update the layout
    fig_wdl.update_layout(title_text='Match outcomes: home vs away')

    ## WLD HEATMAP

    wld_heatmap_data = team_stats['form']

    form_mapping = {'W': 1, 'D': 0, 'L': -1}
    wld_heatmap_data = wld_heatmap_data.apply(lambda x: [form_mapping[i] for i in x])[0]

    # Creating Heatmap
    fig_wld_heatmap = go.Figure(data=go.Heatmap(
        z=[wld_heatmap_data],
        x=list(range(1, len(wld_heatmap_data) + 1)),
        y=[''],
        colorscale=[(0.00, '#DE3E3E'), (0.33, '#DE3E3E'),  # Red for Loss
                    (0.33, '#FDFF6E'), (0.66, '#FDFF6E'),  # Yellow for Draw
                    (0.66, '#61C750'), (1.00, '#61C750')], # Green for Win
        zmin=-1,
        zmax=1,
        hovertemplate='Match no.: %{x}<extra></extra>'
    ))

    # Update layout for heatmap
    fig_wld_heatmap.update_layout(
        xaxis_title='Matches',
        yaxis_title='Results',
        title_text='Season form',
        yaxis=dict(
            title_standoff=25
        ),
    )

    # Explicitly define colorbar with custom labels
    fig_wld_heatmap.update_traces(
        colorbar=dict(
            tickvals=[-1, 0, 1],
            ticktext=['Loss', 'Draw', 'Win'],
            nticks=3
        )
    )


    ## GOALS PLOT
    # Time slots for the goals
    time_slots = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '91-105', '106-120']

    goals_for = []
    goals_against = []

    for slot in time_slots:
        # For goals_for
        goals_for_key = 'goals_for_minute_' + slot + '_total'
        if goals_for_key in team_stats and team_stats[goals_for_key].iloc[0] is not None:
            goals_for.append(team_stats[goals_for_key].iloc[0])
        else:
            goals_for.append(0)  # or some other default value, like None

        # For goals_against
        goals_against_key = 'goals_against_minute_' + slot + '_total'
        if goals_against_key in team_stats and team_stats[goals_against_key].iloc[0] is not None:
            goals_against.append(-team_stats[goals_against_key].iloc[0])
        else:
            goals_against.append(0)  # or some other default value, like None



    # Creating the bar plot
    fig_goals = go.Figure()
    fig_goals.add_trace(go.Bar(x=time_slots, y=goals_for, name='Scored goals', marker_color='#61C750'))
    fig_goals.add_trace(go.Bar(x=time_slots, y=goals_against, name='Conceded goals', marker_color='#DE3E3E')) 

    # Adding title and labels
    fig_goals.update_layout(
        title='Conceded and scored goals',
        xaxis_title='Time Slot',
        yaxis_title='Goal Count',
        barmode='overlay',
        plot_bgcolor='#FFFFFF'
    )

    
    return (logo,
            subtitle,
            {'display': 'block'},
            fig_cards, {"display": "block"}, 
            fig_clean_sheets, {"display": "inline-block"}, 
            fig_wdl, {"display": "inline-block"}, 
            fig_wld_heatmap, {"display": "inline-block"}, 
            fig_goals, {"display": "inline-block"}) 

@app.callback(
    Output('dropdown_regression_input', 'options'),
    Input('dropdown_regression_target', 'value')
)
def updaate_dropdown_regression_options(selected_target):
    if selected_target is None:
        return [{'label': row, 'value': row} for row in regression_columns]
    else:
        options = regression_columns.drop(selected_target)
        return [{'label': row, 'value': row} for row in options]
    
@app.callback(
    Output('regression_matrix', 'figure'),
    Output('regression_matrix', 'style'),
    Input('dropdown_regression_input', 'value'),
    Input('dropdown_regression_target', 'value'),
)
def update_regression_matrix(selected_inputs, selected_target):
    if selected_inputs is None or selected_target is None:
        return ({}, {'display': 'none'})
    else:
        correlation_data = modelgenerator.get_selected_data(selected_inputs + [selected_target])

        # Filter to include only numeric columns
        correlation_data = correlation_data.select_dtypes(include=['number'])

        # Calculate the correlation matrix
        correlation_matrix = correlation_data.corr()

        # Create a list of column names for the labels
        labels = correlation_data.columns

        # Create a heatmap-like plot
        trace = go.Heatmap(z=correlation_matrix.values,
                        x=labels,
                        y=labels,
                        colorscale='Viridis')

        layout = go.Layout(title='Correlation matrix of the selected numeric variables',
                        xaxis=dict(ticks='', side='top'),
                        yaxis=dict(ticks='', side='left'),
                        margin=dict(t=150),
                        height=600,
                        width=800)  # Adjust top margin)

        # Create the figure
        fig_correlation = go.Figure(data=[trace], layout=layout)

        
        return (fig_correlation,
                {'display': 'block'})

@app.callback(
    Output('regression_result', 'children'),
    Input('regression_button', 'n_clicks'),
    dependencies.State('dropdown_regression_input', 'value'),
    dependencies.State('dropdown_regression_target', 'value')
)
def update_regression_result(n_clicks, selected_inputs, selected_target):
    if n_clicks is None or selected_inputs is None or selected_target is None:
        return html.P("Please select inputs and a target variable, then press the button.")

    # Get the selected data
    regression_data = modelgenerator.get_selected_data(selected_inputs)
    target = modelgenerator.get_selected_data([selected_target])

    # Check if the data is sufficient
    if regression_data.empty or target.empty:
        return html.P("Data not available or insufficient for analysis.")

    # Automatically identify and encode categorical columns
    categorical_columns = regression_data.select_dtypes(include=['object', 'category']).columns
    if categorical_columns.any():
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_columns = encoder.fit_transform(regression_data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))
        regression_data = regression_data.drop(categorical_columns, axis=1)
        regression_data = pd.concat([regression_data, encoded_df], axis=1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(regression_data, target, test_size=0.2, random_state=42)

    # Fit the linear regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())

    # Predictions and performance evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Residuals plot - ensuring that y_test and y_test_pred are compatible
    residuals = y_test.to_numpy().flatten() - y_test_pred  # Flatten y_test to match y_test_pred dimensions
    residuals_fig = go.Figure()
    residuals_fig.add_trace(go.Scatter(x=y_test_pred, y=residuals, mode='markers',
                                    name='Residuals'))
    residuals_fig.update_layout(title='Residuals Plot',
                                xaxis_title='Predicted Values',
                                yaxis_title='Residuals',
                                showlegend=True)
    
    # Construct the model formula
    intercept = model.intercept_
    coefficients = model.coef_
    variables = X_train.columns
    model_formula = "y = {:.2f} + ".format(intercept) + " + ".join(["{:.2f} * {}".format(coef, var) for coef, var in zip(coefficients, variables)])

    # Generate a user-friendly explanation of the model
    coef_explanation = ''.join([f'• For every one unit increase in {input_var}, the target variable changes by {coef:.2f} units.\n' 
                                for input_var, coef in zip(X_train.columns, model.coef_)])
    result = html.Div([
        html.H3("Linear Regression Model Results"),
        html.P("Model Equation:"),
        html.Code(model_formula),  # Display the model formula
        html.P(f"Train R² Score: {train_score:.2%}"),
        html.P(f"Test R² Score: {test_score:.2%}"),
        html.P(f"Train RMSE: {train_rmse:.2f}"),
        html.P(f"Test RMSE: {test_rmse:.2f}"),
        html.P("Effects of each input variable on the target:"),
        html.Pre(coef_explanation),
        dcc.Graph(figure=residuals_fig)
    ])

    return result

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)