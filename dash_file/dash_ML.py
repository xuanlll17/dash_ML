from dash import Dash, html, callback, Input, Output, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


dash_ML = Dash(
    requests_pathname_prefix="/dash/ML/", external_stylesheets=[dbc.themes.SANDSTONE]
)
dash_ML.title = "2014-2023信用卡消費樣態"
dataset = pd.read_csv("processed_dataset.csv")
df = pd.DataFrame(dataset)

dash_ML.layout = html.Div(
    [
        dbc.Container(
            [
                html.Div(
                    [
                        html.Div(
                            [html.H1("2014-2023信用卡消費樣態")], className="col text-center"
                        )
                    ],
                    className="row",
                    style={"paddingTop": "1rem"},
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
                                        {"label": "未滿20歲", "value": "未滿20歲"},
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
                                    style={"marginRight": "1rem"},
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("預測月份"),
                                dbc.Select(
                                    id="month",
                                    value="ALL",
                                    options=[
                                        {"label": "10", "value": "10"},
                                        {"label": "11", "value": "11"},
                                        {"label": "12", "value": "12"},
                                        {"label": "ALL", "value": "ALL"},
                                    ],
                                ),
                            ],
                        ),
                    ],
                    className="d-flex justify-content-center",
                    style={"paddingTop": "1rem"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="graph_ind", style={"width": "50%"}),
                        dcc.Graph(id="graph_ar", style={"width": "50%"}),
                        dcc.Graph(id="graph_age", style={"width": "50%"}),
                        dcc.Graph(id="graph_line_age", style={"width": "50%"}),
                        dcc.Graph(id="graph_heatmap_age", style={"width": "50%"}),
                        dcc.Graph(id="graph_heatmap_ind", style={"width": "50%"}),
                        dcc.Graph(id="graph_heatmap_ar", style={"width": "50%"}),
                        dcc.Graph(id="graph_LinearRegression", style={"width": "50%"}),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "justify-content": "center",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.P("來源：聯合信用卡處理中心Open API"),
                            ],
                            className="col",
                        )
                    ],
                    className="row",
                    style={
                        "paddingTop": "2rem",
                        "fontSize": "0.8rem",
                        "lineHeight": "0.3rem",
                    },
                ),
            ],
            style={"maxWidth": "100%", "height": "auto"},
        )
    ],
)


@dash_ML.callback(Output("graph_ind", "figure"), Input("industry", "value"))
def ind_chart(selected_ind):
    global df
    if selected_ind == "ALL":
        agg_df = (
            df.groupby("產業別")
            .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
            .reset_index()
        )
        agg_df["平均交易金額"] = agg_df["信用卡交易金額[新台幣]"] / agg_df["信用卡交易筆數"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=agg_df["產業別"], y=agg_df["信用卡交易金額[新台幣]"], name="信用卡交易金額[新台幣]"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=agg_df["產業別"],
                y=agg_df["平均交易金額"],
                name="平均交易金額",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )
        fig.update_layout(title_text="不同產業別信用卡交易金額及平均交易金額")
        fig.update_xaxes(title_text="產業別")
        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    else:
        agg_df = (
            df.groupby("產業別")
            .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
            .reset_index()
        )
        agg_df["平均交易金額"] = agg_df["信用卡交易金額[新台幣]"] / agg_df["信用卡交易筆數"]

        highlighted_ind = selected_ind
        print(highlighted_ind)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=agg_df["產業別"],
                y=agg_df["信用卡交易金額[新台幣]"],
                name="信用卡交易金額[新台幣]",
                marker_color=[
                    "rgba(1,87,155,0.2)" if ind != highlighted_ind else "blue"
                    for ind in agg_df["產業別"]
                ],
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=agg_df["產業別"],
                y=agg_df["平均交易金額"],
                name="平均交易金額",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text=f"{selected_ind} 信用卡交易金額及平均交易金額")
        fig.update_xaxes(title_text="產業別")
        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)

    return fig


@dash_ML.callback(Output("graph_age", "figure"), Input("age", "value"))
def age_chart(selected_age):
    global df
    if selected_age == "ALL":
        agg_df = (
            df.groupby("年齡層")
            .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
            .reset_index()
        )
        agg_df["平均交易金額"] = agg_df["信用卡交易金額[新台幣]"] / agg_df["信用卡交易筆數"]

        # 將未滿20歲的資料放在第一個
        agg_df = agg_df.sort_values(
            by="年齡層", key=lambda x: x.replace("未滿20歲", "00").replace("20(含)-25歲", "10")
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=agg_df["年齡層"], y=agg_df["信用卡交易金額[新台幣]"], name="信用卡交易金額[新台幣]"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=agg_df["年齡層"],
                y=agg_df["平均交易金額"],
                name="平均交易金額",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text="不同年齡層信用卡交易金額及平均交易金額")

        fig.update_xaxes(title_text="年齡層")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    else:
        agg_df = (
            df.groupby("年齡層")
            .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
            .reset_index()
        )
        agg_df["平均交易金額"] = agg_df["信用卡交易金額[新台幣]"] / agg_df["信用卡交易筆數"]

        # 將未滿20歲的資料放在第一個
        agg_df = agg_df.sort_values(
            by="年齡層", key=lambda x: x.replace("未滿20歲", "00").replace("20(含)-25歲", "10")
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        highlighted_age = selected_age

        fig.add_trace(
            go.Bar(
                x=agg_df["年齡層"],
                y=agg_df["信用卡交易金額[新台幣]"],
                name="信用卡交易金額[新台幣]",
                marker_color=[
                    "rgba(1,87,155,0.2)" if age != highlighted_age else "blue"
                    for age in agg_df["年齡層"]
                ],
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=agg_df["年齡層"],
                y=agg_df["平均交易金額"],
                name="平均交易金額",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text=f"{selected_age} 信用卡交易金額及平均交易金額")

        fig.update_xaxes(title_text="年齡層")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    return fig


@dash_ML.callback(Output("graph_ar", "figure"), Input("area", "value"))
def ar_chart(selected_ar):
    global df
    if selected_ar == "ALL":
        agg_df = (
            df.groupby("地區")
            .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
            .reset_index()
        )

        agg_df["平均交易金額"] = agg_df["信用卡交易金額[新台幣]"] / agg_df["信用卡交易筆數"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=agg_df["地區"], y=agg_df["信用卡交易金額[新台幣]"], name="信用卡交易金額[新台幣]"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=agg_df["地區"],
                y=agg_df["平均交易金額"],
                name="平均交易金額",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text="不同地區信用卡交易金額及平均交易金額")

        fig.update_xaxes(title_text="地區")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    else:
        agg_df = (
            df.groupby("地區")
            .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
            .reset_index()
        )
        agg_df["平均交易金額"] = agg_df["信用卡交易金額[新台幣]"] / agg_df["信用卡交易筆數"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        highlighted_ar = selected_ar

        fig.add_trace(
            go.Bar(
                x=agg_df["地區"],
                y=agg_df["信用卡交易金額[新台幣]"],
                name="信用卡交易金額[新台幣]",
                marker_color=[
                    "rgba(1,87,155,0.2)" if ar != highlighted_ar else "blue"
                    for ar in agg_df["地區"]
                ],
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=agg_df["地區"],
                y=agg_df["平均交易金額"],
                name="平均交易金額",
                mode="lines+markers",
                marker=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text=f"{selected_ar} 信用卡交易金額及平均交易金額")

        fig.update_xaxes(title_text="地區")

        fig.update_yaxes(title_text="信用卡交易金額[新台幣]", secondary_y=False)
        fig.update_yaxes(title_text="平均交易金額", secondary_y=True)
    return fig


@dash_ML.callback(Output("graph_line_age", "figure"), Input("age", "value"))
def lineAge_chart(selected_ar):
    global df
    if selected_ar == "ALL":
        year_total = df.groupby(["年", "年齡層"])["信用卡交易金額[新台幣]"].sum().reset_index()
        fig = px.line(
            year_total,
            x="年",
            y="信用卡交易金額[新台幣]",
            color="年齡層",
            title="不同年齡層每年信用卡交易金額趨勢",
            markers=True,
        )
    else:
        year_total = df.groupby(["年", "年齡層"])["信用卡交易金額[新台幣]"].sum().reset_index()
        filtered_df = year_total[year_total["年齡層"] == f"{selected_ar}"]
        fig = px.line(
            filtered_df,
            x="年",
            y="信用卡交易金額[新台幣]",
            color="年齡層",
            title=f"{selected_ar} 每年信用卡交易金額趨勢",
            markers=True,
        )
    return fig


@dash_ML.callback(
    Output("graph_heatmap_age", "figure"), Input("graph_heatmap_age", "id")
)
def heatmapAge_chart(graph_id):
    global df

    grouped_data = (
        df.groupby(["年", "年齡層"])
        .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
        .reset_index()
    )

    grouped_data["平均交易金額"] = grouped_data["信用卡交易金額[新台幣]"] / grouped_data["信用卡交易筆數"]

    # 修改排序方式，確保未滿20歲顯示在最前面
    age_order = ["未滿20歲"] + sorted(set(grouped_data["年齡層"]) - {"未滿20歲"})

    grouped_data["年齡層"] = pd.Categorical(
        grouped_data["年齡層"], categories=age_order, ordered=True
    )

    pivot_table_year = grouped_data.pivot_table(
        index="年", columns="年齡層", values="平均交易金額", aggfunc="mean"
    )

    pivot_table_year.reset_index(inplace=True)

    print(pivot_table_year)

    fig = px.imshow(
        pivot_table_year.set_index("年"),
        labels=dict(x="年齡層", y="年", color="平均交易金額"),
        color_continuous_scale="viridis",
        text_auto=True,
    )
    fig.update_layout(title="年 / 年齡層 平均交易金額熱力圖")
    return fig


@dash_ML.callback(
    Output("graph_heatmap_ind", "figure"), Input("graph_heatmap_ind", "id")
)
def heatmapInd_chart(graph_id):
    global df

    grouped_data = (
        df.groupby(["產業別", "年齡層"])
        .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
        .reset_index()
    )

    grouped_data["平均交易金額"] = grouped_data["信用卡交易金額[新台幣]"] / grouped_data["信用卡交易筆數"]

    # 修改排序方式，確保未滿20歲顯示在最前面
    age_order = ["未滿20歲"] + sorted(set(grouped_data["年齡層"]) - {"未滿20歲"})

    grouped_data["年齡層"] = pd.Categorical(
        grouped_data["年齡層"], categories=age_order, ordered=True
    )

    pivot_table_year = grouped_data.pivot_table(
        index="產業別", columns="年齡層", values="平均交易金額", aggfunc="mean"
    )

    pivot_table_year.reset_index(inplace=True)

    fig = px.imshow(
        pivot_table_year.set_index("產業別"),
        labels=dict(x="產業別", y="年齡層", color="平均交易金額"),
        color_continuous_scale="viridis",
        text_auto=True,
    )
    fig.update_layout(title="產業別 / 年齡層 平均交易金額熱力圖")
    return fig


@dash_ML.callback(Output("graph_heatmap_ar", "figure"), Input("graph_heatmap_ar", "id"))
def heatmapAr_chart(graph_id):
    global df

    grouped_data = (
        df.groupby(["地區", "年齡層"])
        .agg({"信用卡交易金額[新台幣]": "sum", "信用卡交易筆數": "sum"})
        .reset_index()
    )

    grouped_data["平均交易金額"] = grouped_data["信用卡交易金額[新台幣]"] / grouped_data["信用卡交易筆數"]

    # 修改排序方式，確保未滿20歲顯示在最前面
    age_order = ["未滿20歲"] + sorted(set(grouped_data["年齡層"]) - {"未滿20歲"})

    grouped_data["年齡層"] = pd.Categorical(
        grouped_data["年齡層"], categories=age_order, ordered=True
    )

    pivot_table_year = grouped_data.pivot_table(
        index="地區", columns="年齡層", values="平均交易金額", aggfunc="mean"
    )

    pivot_table_year.reset_index(inplace=True)

    fig = px.imshow(
        pivot_table_year.set_index("地區"),
        labels=dict(x="年齡層", y="地區", color="平均交易金額"),
        color_continuous_scale="viridis",
        text_auto=True,
    )
    fig.update_layout(title="地區 / 年齡層 平均交易金額熱力圖")
    return fig


@dash_ML.callback(Output("graph_LinearRegression", "figure"), Input("month", "value"))
def LinearRegression_chart(selected_mon):
    global df
    df["年月"] = pd.to_datetime(df["年"].astype(str) + df["月"].astype(str), format="%Y%m")
    df["年月"] = df["年月"].dt.strftime("%Y%m")

    monthly_total_expenses = df.groupby(["年月"])["信用卡交易金額[新台幣]"].sum().reset_index()

    monthly_total_expenses["年份"] = pd.to_datetime(
        monthly_total_expenses["年月"], format="%Y%m"
    ).dt.year
    monthly_total_expenses["月份"] = pd.to_datetime(
        monthly_total_expenses["年月"], format="%Y%m"
    ).dt.month

    X = monthly_total_expenses[["年份", "月份"]].astype(float)
    y = monthly_total_expenses["信用卡交易金額[新台幣]"]

    degree = 2
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_poly)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    model_scaled = LinearRegression()
    model_scaled.fit(X_scaled, y_scaled)

    if selected_mon == "ALL":
        next_months = pd.DataFrame({"年份": [2023] * 3, "月份": [10, 11, 12]})
        next_months_poly = poly.transform(next_months)
        next_months_scaled = scaler_X.transform(next_months_poly)
        next_months["預測信用卡金額_scaled"] = scaler_y.inverse_transform(
            model_scaled.predict(next_months_scaled).reshape(-1, 1)
        ).flatten()

        r_squared_scaled = r2_score(y_scaled, model_scaled.predict(X_scaled))
        print(f"R-squared value (scaled): {r_squared_scaled}")
        mse_scaled = mean_squared_error(y_scaled, model_scaled.predict(X_scaled))
        print(f"均方差 (scaled): {mse_scaled}")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_total_expenses["年月"], y=y, mode="markers", name="實際信用卡金額"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=next_months["年份"].astype(str)
                + next_months["月份"].astype(str).str.zfill(2),
                y=next_months["預測信用卡金額_scaled"],
                mode="markers+lines",
                name="預測信用卡金額",
                line=dict(dash="dash", color="red"),
            )
        )

        fig.update_layout(
            title="每月信用卡金額及預測",
            xaxis_title="年月",
            yaxis_title="信用卡交易金額",
            xaxis=dict(tickmode="linear", tick0=0, dtick=8),
            showlegend=True,
        )
    else:
        next_months = pd.DataFrame({"年份": 2023, "月份": [selected_mon]})
        next_months_poly = poly.transform(next_months)
        next_months_scaled = scaler_X.transform(next_months_poly)
        next_months["預測信用卡金額_scaled"] = scaler_y.inverse_transform(
            model_scaled.predict(next_months_scaled).reshape(-1, 1)
        ).flatten()

        r_squared_scaled = r2_score(y_scaled, model_scaled.predict(X_scaled))
        print(f"R-squared value (scaled): {r_squared_scaled}")
        mse_scaled = mean_squared_error(y_scaled, model_scaled.predict(X_scaled))
        print(f"均方差 (scaled): {mse_scaled}")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_total_expenses["年月"], y=y, mode="markers", name="實際信用卡金額"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=next_months["年份"].astype(str)
                + next_months["月份"].astype(str).str.zfill(2),
                y=next_months["預測信用卡金額_scaled"],
                mode="markers+lines",
                name="預測信用卡金額",
                line=dict(dash="dash", color="red"),
            )
        )

        fig.update_layout(
            title=f"每月信用卡金額及{selected_mon}月預測",
            xaxis_title="年月",
            yaxis_title="信用卡交易金額",
            xaxis=dict(tickmode="linear", tick0=0, dtick=8),
            showlegend=True,
        )

    return fig
