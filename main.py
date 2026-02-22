from flask import Flask, render_template, request, redirect
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

FILE = "expenses.csv"

# --------------------------------
# Create CSV if not exists
# --------------------------------
if not os.path.exists(FILE):
    df = pd.DataFrame(columns=["date", "category", "amount"])
    df.to_csv(FILE, index=False)


# --------------------------------
# HOME ROUTE
# --------------------------------
@app.route("/")
def home():
    df = pd.read_csv(FILE)

    if df.empty:
        return render_template(
            "index.html",
            data=[],
            categories=[],
            total=0,
            graphJSON=None,
            futureGraphJSON=None
        )

    df["date"] = pd.to_datetime(df["date"])

    total = df["amount"].sum()
    categories = df["category"].unique()

    # -------------------------------
    # CATEGORY TOTAL GRAPH
    # -------------------------------
    category_totals = df.groupby("category")["amount"].sum()

    graph = go.Bar(
        x=category_totals.index,
        y=category_totals.values
    )

    graphJSON = json.dumps([graph], cls=plotly.utils.PlotlyJSONEncoder)

    # -------------------------------
    # AI FUTURE PREDICTION
    # -------------------------------
    df_sorted = df.sort_values("date")
    df_sorted["day_number"] = (df_sorted["date"] - df_sorted["date"].min()).dt.days

    X = df_sorted[["day_number"]]
    y = df_sorted["amount"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = 7
    last_day = df_sorted["day_number"].max()

    future_X = np.array([[last_day + i] for i in range(1, future_days + 1)])
    future_predictions = model.predict(future_X)

    future_graph = go.Bar(
        x=[f"Day {i}" for i in range(1, future_days + 1)],
        y=future_predictions
    )

    futureGraphJSON = json.dumps([future_graph], cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "index.html",
        data=df.to_dict(orient="records"),
        categories=categories,
        total=total,
        graphJSON=graphJSON,
        futureGraphJSON=futureGraphJSON
    )


# --------------------------------
# ADD EXPENSE
# --------------------------------
@app.route("/add", methods=["POST"])
def add():
    category = request.form["category"]
    amount = float(request.form["amount"])
    date = datetime.now().strftime("%Y-%m-%d")

    new_data = pd.DataFrame([[date, category, amount]],
                            columns=["date", "category", "amount"])

    new_data.to_csv(FILE, mode="a", header=False, index=False)

    return redirect("/")


# --------------------------------
# DELETE EXPENSE
# --------------------------------
@app.route("/delete/<int:index>")
def delete(index):
    df = pd.read_csv(FILE)
    df = df.drop(index)
    df.to_csv(FILE, index=False)
    return redirect("/")


# --------------------------------
# RUN
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
