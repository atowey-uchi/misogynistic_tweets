#!/usr/bin/env python
# coding: utf-8

import base64
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
import dash
from dash import dash_table, dcc, html

warnings.filterwarnings("ignore")


# # Data Cleaning and Organization

# Read in data about candidates (including state, position, twitter username,
# ideology and leadership scores, etc).


df = pd.read_csv("./Data/pred.csv")
df = df[df["Predicted"] == 0]
df = df[["candidate_user_name", "pol_party"]]
df["Count"] = df.groupby(["candidate_user_name"])["pol_party"].transform("count")
df = df.drop_duplicates()


# Read in data from misogynistic tweets from our Classifier.


miso_df = pd.read_csv("./Data/misogynistic_tweets.csv")
miso_df = miso_df[
    [
        "Full Name",
        "candidate_user_name",
        "party",
        "ideology",
        "leadership",
        "state",
        "Position",
    ]
]
df = df.merge(miso_df, on="candidate_user_name")


# Merge data and count number of tweets.


df = df[
    [
        "Full Name",
        "state",
        "party",
        "Position",
        "candidate_user_name",
        "ideology",
        "leadership",
        "Count",
    ]
]
df = df.sort_values(["Count"], ascending=False)
df["ideology"] = df["ideology"].round(decimals=3)
df["leadership"] = df["leadership"].round(decimals=3)
df = df.groupby(
    ["Full Name", "state", "party", "Position", "ideology", "leadership"],
    as_index=False,
).agg({"Count": "sum"})


# Organize data to use in Plotly histograms.


df_log = df.copy()
df_log["Count_l"] = np.log(df_log["Count"])
df_log.head()


# Count the number of tweets (logged) by State to display in Chloropleth map
# in Dash.


state_df = df_log[["state", "Count", "Count_l"]]
state_df = state_df.groupby(["state"]).sum()
state_df = state_df.reset_index()
state_df.head()


# Then, for display purposes, we updated the column headings for aesthetic and
# readability purposes. The following data table will be included in the Dash.


df = df.rename(
    columns={
        "state": "State",
        "party": "Party",
        "ideology": "Ideology",
        "leadership": "Leadership",
    }
)
df.head()


# # Plotly and Dash

# ### Initalize Dash, set up server, and create application-wide variables.


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)
app.title = "Women In Politics & Misogynistic Tweets"
server = app.server

IMAGE = "./assets/wordcloud.png"
encoded_image = base64.b64encode(open(IMAGE, "rb").read())

PAGE_SIZE = 10


# ### Create Figures Using Plotly

# Note that the output in this Jupyter Notebook uses light text colors because
# the Dash has a dark background color. For maximum readibility, please view
# plots in the Dash (https://twittermisogyny.herokuapp.com/).

# Figure 1: A Histogram Comparing Number of Tweets (Logged) to Ideology Score.


fig = px.histogram(
    df_log,
    x="ideology",
    y="Count",
    log_y=True,
    range_x=[-0.1, 1.1],
    hover_name="Full Name",
    labels=dict(ideology="Ideology Score", Count="Number of Tweets"),
)

fig.update_xaxes(
    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
    tickfont=dict(color="white", size=12, family="Helvetica"),
    title_font=dict(size=14, color="#dadfeb", family="Helvetica"),
    showgrid=False,
)


fig.update_yaxes(
    title_text="Number of Tweets",
    title_font=dict(size=14, color="#dadfeb", family="Helvetica"),
    showgrid=True,
    tickfont=dict(color="#dadfeb", size=12, family="Helvetica"),
)

fig.update_layout(
    title_font_family="Helvetica",
    title_font_color="#dadfeb",
    font=dict(size=14),
    title={
        "text": "Number of Tweets By Ideology Score of Politician",
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
)

fig.update_layout(
    {
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "yaxis": {"gridcolor": "#0c2b5c"},
    }
)

fig.update_traces(marker_color="#2e75e6")


# Figure 2: A Histogram Comparing Number of Tweets (Logged) to Leadership
# Score.


fig2 = px.histogram(
    df_log,
    x="leadership",
    y="Count",
    log_y=True,
    range_x=[-0.1, 1.1],
    hover_name="Full Name",
    labels=dict(leadership="Leadership Score", Count="Number of Tweets"),
)

fig2.update_yaxes(
    title_text="Number of Tweets",
    title_font=dict(size=14, color="#dadfeb", family="Helvetica"),
    showgrid=True,
    tickfont=dict(family="Helvetica", color="#dadfeb", size=12),
)

fig2.update_xaxes(
    range=[0, 1.0001],
    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
    autorange=False,
    showgrid=False,
    tickfont=dict(family="Helvetica", color="#dadfeb", size=12),
    title_font=dict(size=14, color="#dadfeb", family="Helvetica"),
)

fig2.update_layout(
    title={
        "text": "Number of Tweets By Leadership Score of Politician",
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    title_font_family="Helvetica",
    title_font_color="#dadfeb",
    font=dict(size=14),
)

fig2.update_layout(
    {
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "yaxis": {"gridcolor": "#0c2b5c"},
    }
)

fig2.update_traces(marker_color="#2e75e6")


# Figure 3: Chloropleth Map for Number of Tweets (logged) By State


fig3 = px.choropleth(
    state_df,
    locations="state",
    color="Count_l",
    color_continuous_scale="blues",
    hover_name="state",
    hover_data=["Count"],
    locationmode="USA-states",
    labels={"Tweets per State"},
    scope="usa",
)

fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=False)

fig3.update_layout(
    title={
        "text": "Number of Tweets By State of Politician",
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    title_font_family="Helvetica",
    title_font_color="#dadfeb",
    font=dict(size=14),
    coloraxis_colorbar=dict(
        title="",
        tickvals=[0.2, 5.481784],
        tickfont={"color": "#dadfeb"},
        ticktext=["Least Tweets", "Most Tweets"],
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_showscale=False,
)

fig3.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"}
)

fig3.update_geos(bgcolor="rgba(0, 0, 0, 0)", showlakes=False)


# ### Set up Dash layout and deploy to server


app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Women in Politics and Misogynistic Tweets: \
                        Classifying Misogyny"
                ),
                html.H2(
                    children="In the run up to the 2020 election, to what \
                        extent is misogynistic rhetoric directed at women \
                            running for office on Twitter in the United \
                                States?"
                ),
            ],
            className="app__header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                dcc.Graph(id="ideology", figure=fig),
                                html.P(
                                    children="Ideology Scores from GovTrack \
                                        USA. Scale 0: Most Liberal to 1: Most \
                                            Conservative"
                                ),
                            ],
                            className="half graph-container",
                        ),
                        html.Div(
                            children=[
                                dcc.Graph(id="map", figure=fig3),
                                html.P(
                                    children="Darker colored states denote \
                                        states with politicians with more \
                                            tweets identified."
                                ),
                            ],
                            className="half graph-container",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                dcc.Graph(id="leadership", figure=fig2),
                                html.P(
                                    children="Leadership Scores from GovTrack \
                                        USA. Scale 0: Least Likely to Sponsor \
                                            Legislation/Hold Leadership Roles \
                                                to 1: Most Likely"
                                ),
                            ],
                            className="half graph-container",
                        ),
                        html.Div(
                            children=[
                                html.H3(
                                    children="Most Commonly Found Words in \
                                        Misogynistic Tweets"
                                ),
                                html.Img(
                                    src="data:image/png;base64,{}".format(
                                        encoded_image.decode()
                                    ),
                                    style={"text-align": "center",
                                           "max-width": "100%"},
                                ),
                            ],
                            className="half",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                dash_table.DataTable(
                                    id="table-paging-and-sorting",
                                    columns=[
                                        {"name": i, "id": i, "deletable": True}
                                        for i in sorted(df.columns)
                                    ],
                                    style_cell={
                                        "padding": "5px",
                                        "fontSize": 16,
                                        "font-family": "Helvetica",
                                        "backgroundColor": "#0c2b5c",
                                    },
                                    style_header={"backgroundColor": "#224f99",},
                                    data=df.to_dict("records"),
                                    style_data={"border": "1px solid #dadfeb"},
                                    page_current=0,
                                    page_size=PAGE_SIZE,
                                    page_action="custom",
                                    sort_action="custom",
                                    sort_mode="single",
                                    sort_by=[],
                                )
                            ],
                            className="full",
                        )
                    ],
                    className="row",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(
    Output("table-paging-and-sorting", "data"),
    Input("table-paging-and-sorting", "page_current"),
    Input("table-paging-and-sorting", "page_size"),
    Input("table-paging-and-sorting", "sort_by"),
)
def update_table(page_current, page_size, sort_by):
    '''
    User-defined filtration of data table
    '''
    if len(sort_by):
        dff = df.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )
    else:
        # No sort is applied
        dff = df

    return dff.iloc[page_current * page_size : (page_current + 1) * page_size].to_dict(
        "records"
    )


if __name__ == "__main__":
    app.run_server()
