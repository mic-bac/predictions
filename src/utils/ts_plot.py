import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_time_series(x, y, yhat, yhat_lower=None, yhat_upper=None, title="Time Series", y_label="Value", vline_x=None):
    """
    Plots original and predicted time series data, optionally with a confidence interval and a vertical line.

    Parameters:
        x (array-like): Time points.
        y (array-like): Original y values.
        yhat (array-like): Predicted y values.
        yhat_lower (array-like, optional): Lower bound of the confidence interval.
        yhat_upper (array-like, optional): Upper bound of the confidence interval.
        title (str, optional): Plot title.
        y_label (str, optional): Y axis label.
        vline_x (float or str, optional): The x value at which to draw a vertical red line.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    ax.plot(x, y, 'o-', label='Actual', color='blue', linewidth=2, markersize=4)
    
    # Plot predicted values
    ax.plot(x, yhat, 'o-', label='Predicted', color='orange', linewidth=2, markersize=4)
    
    # Add confidence interval if provided
    if yhat_lower is not None and yhat_upper is not None:
        ax.fill_between(x, yhat_lower, yhat_upper, alpha=0.2, color='orange', label='Confidence Interval')
    
    # Add vertical line if vline_x is specified
    if vline_x is not None:
        ax.axvline(x=vline_x, color='red', linestyle='--', linewidth=2)
        ax.text(vline_x, ax.get_ylim()[1] * 0.95, 'Cut-off', rotation=0, ha='right', va='top', fontsize=10, color='red')
    
    # Labels and formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis for dates if applicable
    try:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=45, ha='right')
    except:
        pass
    
    plt.tight_layout()
    plt.show()