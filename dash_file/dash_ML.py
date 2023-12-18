from dash import Dash, html, dash_table, callback, Input, Output, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dash_ML = Dash(
    requests_pathname_prefix="/dash/ML/", external_stylesheets=[dbc.themes.BOOTSTRAP]
)
dash_ML.title = "信用卡消費樣態"
dataset = pd.read_csv("../processed_dataset.csv")
df = pd.DataFrame(dataset)

dash_ML.layout = html.Div(
    [
        dbc.Container(
            [
                html.Div(
                    [html.Div([html.H1("信用卡消費樣態")], className="col text-center")],
                    className="row",
                    style={"paddingTop": "3rem"},
                ),
                 html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("地區"),
                                dbc.Select(
                                    id="area",
                                    value="ALL",
                                    options=[
                                        {"label": "臺北市", "value": "臺北市"},
                                        {"label": "新北市", "value": "新北市"},
                                        {"label": "桃園市", "value": "桃園市"},
                                        {"label": "臺中市", "value": "臺中市"},
                                        {"label": "臺南市", "value": "臺南市"},
                                        {"label": "高雄市", "value": "高雄市"},
                                        {"label": "ALL", "value": "ALL"},
                                    ],
                                    style={"marginRight": "1rem"},
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("產業別"),
                                dbc.Select(
                                    id="industry",
                                    value="ALL",
                                    options=[
                                        {"label": "食", "value": "食"},
                                        {"label": "衣", "value": "衣"},
                                        {"label": "住", "value": "住"},
                                        {"label": "行", "value": "行"},
                                        {"label": "文教康樂", "value": "文教康樂"},
                                        {"label": "百貨", "value": "百貨"},
                                        {"label": "ALL", "value": "ALL"},
                                    ],
                                    style={"marginRight": "1rem"},
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("年齡層"),
                                dbc.Select(
                                    id="age",
                                    value="ALL",
                                    options=[
                                        {"label": "20(含)-25歲", "value": "20(含)-25歲"},
                                        {"label": "25(含)-30歲", "value": "25(含)-30歲"},
                                        {"label": "30(含)-35歲", "value": "30(含)-35歲"},
                                        {"label": "35(含)-40歲", "value": "35(含)-40歲"},
                                        {"label": "40(含)-45歲", "value": "40(含)-45歲"},
                                        {"label": "45(含)-50歲", "value": "45(含)-50歲"},
                                        {"label": "50(含)-55歲", "value": "50(含)-55歲"},
                                        {"label": "55(含)-60歲", "value": "55(含)-60歲"},
                                        {"label": "60(含)-65歲", "value": "60(含)-65歲"},
                                        {"label": "70(含)-75歲", "value": "70(含)-75歲"},
                                        {"label": "75(含)-80歲", "value": "75(含)-80歲"},
                                        {"label": "80(含)歲以上", "value": "80(含)歲以上"},
                                        {"label": "ALL", "value": "ALL"},
                                    ],
                                ),
                            ],
                        )
                    ],
                    className="d-flex justify-content-center",
                    style={"paddingTop": "2rem"},
                ),
                html.Div([
                    dcc.Graph(id="graph_line", style={'flex': '3'}),
                    dcc.Graph(id="graph_age", style={'flex': '3'}),
                    dcc.Graph(id="graph_ar", style={'flex': '3'}),
                    dcc.Graph(id="graph_line_age", style={'flex': '3'}),
                ],style={'display': 'flex', 'flexWrap': 'wrap'}),
            ],
            style={"maxWidth":"100%","height":"auto"}
        )
    ],
    
)

@dash_ML.callback(
    Output("graph_line", "figure"),
    Input("industry", "value")
)
def line_chart(selected_ind):
    global df
    if selected_ind == "ALL":
        df['平均交易金額'] = df['信用卡交易金額[新台幣]'] / df['信用卡交易筆數']
        agg_df = df.groupby('產業別').agg({
            '信用卡交易金額[新台幣]': 'sum',
            '信用卡交易筆數': 'sum',
            '平均交易金額': 'mean'
        }).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=agg_df['產業別'], y=agg_df['信用卡交易金額[新台幣]'], name="信用卡交易金額[新台幣]"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=agg_df['產業別'], y=agg_df['平均交易金額'], name="平均交易金額", mode='lines+markers', marker=dict(color='red')),
            secondary_y=True,
        )
        fig.update_layout(
            title_text="不同產業別的信用卡交易金額及平均交易金額"
        )
        fig.update_xaxes(title_text="產業別")
        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    else:
        # Handle the case when a specific industry is selected
        df['平均交易金額'] = df['信用卡交易金額[新台幣]'] / df['信用卡交易筆數']
        agg_df = df.groupby('產業別').agg({
            '信用卡交易金額[新台幣]': 'sum',
            '信用卡交易筆數': 'sum',
            '平均交易金額': 'mean'
        }).reset_index()

        # Set color to blue for the selected industry
        highlighted_ind = selected_ind
        print(highlighted_ind)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=agg_df['產業別'], y=agg_df['信用卡交易金額[新台幣]'], name="信用卡交易金額[新台幣]", marker_color=['rgba(1,87,155,0.2)' if ind != highlighted_ind else 'blue' for ind in agg_df['產業別']]),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=agg_df['產業別'], y=agg_df['平均交易金額'], name="平均交易金額", mode='lines+markers', marker=dict(color='red')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="不同產業別的信用卡交易金額及平均交易金額"
        )
        fig.update_xaxes(title_text="產業別")
        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)

    return fig

@dash_ML.callback(
    Output("graph_age", "figure"),
    Input("age", "value")
)
def line_chart(selected_age):
    global df
    if selected_age == "ALL":
        df['平均交易金額'] = df['信用卡交易金額[新台幣]'] / df['信用卡交易筆數']

        agg_df = df.groupby('年齡層').agg({
            '信用卡交易金額[新台幣]': 'sum',
            '信用卡交易筆數': 'sum',
            '平均交易金額': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=agg_df['年齡層'], y=agg_df['信用卡交易金額[新台幣]'], name="信用卡交易金額[新台幣]"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=agg_df['年齡層'], y=agg_df['平均交易金額'], name="平均交易金額", mode='lines+markers', marker=dict(color='red')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="不同年齡層的信用卡交易金額及平均交易金額"
        )

        fig.update_xaxes(title_text="年齡層")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    else:
        df['平均交易金額'] = df['信用卡交易金額[新台幣]'] / df['信用卡交易筆數']

        agg_df = df.groupby('年齡層').agg({
            '信用卡交易金額[新台幣]': 'sum',
            '信用卡交易筆數': 'sum',
            '平均交易金額': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        highlighted_age = selected_age

        fig.add_trace(
            go.Bar(x=agg_df['年齡層'], y=agg_df['信用卡交易金額[新台幣]'], name="信用卡交易金額[新台幣]", marker_color=['rgba(1,87,155,0.2)' if age != highlighted_age else 'blue' for age in agg_df['年齡層']]),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=agg_df['年齡層'], y=agg_df['平均交易金額'], name="平均交易金額", mode='lines+markers', marker=dict(color='red')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="不同年齡層的信用卡交易金額及平均交易金額"
        )

        fig.update_xaxes(title_text="年齡層")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    return fig

@dash_ML.callback(
    Output("graph_ar", "figure"),
    Input("area", "value")
)
def line_chart(selected_ar):
    global df
    if selected_ar == "ALL":
        df['平均交易金額'] = df['信用卡交易金額[新台幣]'] / df['信用卡交易筆數']

        agg_df = df.groupby('地區').agg({
            '信用卡交易金額[新台幣]': 'sum',
            '信用卡交易筆數': 'sum',
            '平均交易金額': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=agg_df['地區'], y=agg_df['信用卡交易金額[新台幣]'], name="信用卡交易金額[新台幣]"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=agg_df['地區'], y=agg_df['平均交易金額'], name="平均交易金額", mode='lines+markers', marker=dict(color='red')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="不同地區的信用卡交易金額及平均交易金額"
        )

        fig.update_xaxes(title_text="地區")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    else:
        df['平均交易金額'] = df['信用卡交易金額[新台幣]'] / df['信用卡交易筆數']

        agg_df = df.groupby('地區').agg({
            '信用卡交易金額[新台幣]': 'sum',
            '信用卡交易筆數': 'sum',
            '平均交易金額': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        highlighted_ar = selected_ar

        fig.add_trace(
            go.Bar(x=agg_df['地區'], y=agg_df['信用卡交易金額[新台幣]'], name="信用卡交易金額[新台幣]", marker_color=['rgba(1,87,155,0.2)' if ar != highlighted_ar else 'blue' for ar in agg_df['地區']]),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=agg_df['地區'], y=agg_df['平均交易金額'], name="平均交易金額", mode='lines+markers', marker=dict(color='red')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="不同地區的信用卡交易金額及平均交易金額"
        )

        fig.update_xaxes(title_text="地區")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    return fig

@dash_ML.callback(
    Output("graph_line_age", "figure"),
    Input("age", "value")
)
def line_chart(selected_ar):
    global df
    if selected_ar == "ALL":
        year_total = df.groupby(['年', '年齡層'])['信用卡交易金額[新台幣]'].sum().reset_index()
        fig = px.line(year_total, x="年", y="信用卡交易金額[新台幣]", color="年齡層", title='各年齡層每年信用卡交易金額趨勢', markers=True)
    else:
        year_total = df.groupby(['年', '年齡層'])['信用卡交易金額[新台幣]'].sum().reset_index()
        filtered_df = year_total[year_total['年齡層'] == f'{selected_ar}']
        fig = px.line(filtered_df, x="年", y="信用卡交易金額[新台幣]", color="年齡層", title=f'{selected_ar}每年信用卡交易金額趨勢', markers=True)
    return fig