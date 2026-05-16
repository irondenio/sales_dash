import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def descriptive(df: pd.DataFrame):
    """Return basic descriptive statistics and a summary table.
    Returns a dict with keys:
        - 'stats': DataFrame of describe()
        - 'fig': Plotly table figure
    """
    stats = df.describe(include='all').reset_index()
    fig = go.Figure(data=[go.Table(header=dict(values=list(stats.columns)),
                                 cells=dict(values=[stats[col] for col in stats.columns]))])
    return {"stats": stats, "fig": fig}


def diagnostic(df: pd.DataFrame, target: str):
    """Identify correlations between features and target.
    Returns a dict with correlation heatmap figure.
    """
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    return {"corr": corr, "fig": fig}


def exploratory(df: pd.DataFrame):
    """Generate exploratory visualizations (distribution, trends).
    Returns a list of Plotly figures.
    """
    figs = []
    for col in df.select_dtypes(include=[np.number]).columns:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        figs.append(fig)
    return {"figs": figs}


def inferential(df: pd.DataFrame, target: str):
    """Perform simple inferential analysis (ANOVA-like) using groupby stats.
    Returns a dict with group statistics and a bar chart.
    """
    groups = df.groupby(target).mean().reset_index()
    fig = px.bar(groups, x=target, y=groups.select_dtypes(include=[np.number]).columns.tolist(), barmode='group')
    return {"group_stats": groups, "fig": fig}


def predictive(df: pd.DataFrame, target: str, model):
    """Fit provided model, return predictions and performance metrics.
    Expects model already fitted on training data.
    """
    X = df.drop(columns=[target])
    y_true = df[target]
    y_pred = model.predict(X)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    return {"y_pred": y_pred, "metrics": metrics}


def prescriptive(df: pd.DataFrame, target: str, model):
    """Generate actionable recommendations based on model feature importance.
    Returns a dict with top features and simple suggestions.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        importances = np.zeros(df.shape[1]-1)
    feature_names = df.drop(columns=[target]).columns
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    recommendations = [f"Focus on improving {feat} (importance={imp:.2f})" for feat, imp in top_features]
    return {"top_features": top_features, "recommendations": recommendations}
