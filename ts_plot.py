import plotly.graph_objs as go

def plot_time_series(x, y, yhat, yhat_lower=None, yhat_upper=None, title="Time Series", y_label="Value"):
    """
    Plots original and predicted time series data, optionally with a confidence interval.

    Parameters:
        x (array-like): Time points.
        y (array-like): Original y values.
        yhat (array-like): Predicted y values.
        yhat_lower (array-like, optional): Lower bound of the confidence interval.
        yhat_upper (array-like, optional): Upper bound of the confidence interval.
        title (str, optional): Plot title.
        y_label (str, optional): Y axis label.
    """
    fig = go.Figure()

    # Original values
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=x, y=yhat,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='orange')
    ))

    # Add confidence interval if provided
    if yhat_lower is not None and yhat_upper is not None:
        fig.add_trace(go.Scatter(
            x=x.tolist() + x[::-1].tolist(),
            y=yhat_upper.tolist() + yhat_lower[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        template="plotly_white"
    )

    fig.show()
