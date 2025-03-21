# BASE
# ------------------------------------------------------
import numpy as np
import pandas as pd
import os
import gc
import warnings
import calendar
import plotly.io as pio

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# CONFIGURATIONS
# ------------------------------------------------------
#pd.set_option('display.max_columns', None)
#pd.options.display.float_format = '{:.2f}'.format
#warnings.filterwarnings('ignore')


train = pd.read_csv("./dataset/train.csv")
test = pd.read_csv("./dataset/test.csv")
stores = pd.read_csv("./dataset/stores.csv")
submission = pd.read_csv("./dataset/sample_submission.csv")   
transactions = pd.read_csv("./dataset/transactions.csv")#.sort_values(["store_nbr", "date"])
holidays = pd.read_csv("./dataset/holidays_events.csv")
oil = pd.read_csv("./dataset/oil.csv")


################################################################
############### Train(Sales) ##################

train['date'] = pd.to_datetime(train['date'])
train['onpromotion'] = train['onpromotion'].astype('float16')
train['sales'] = train['sales'].astype('float32')

train['day_of_week'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

data_grouped_day = train.groupby(['day_of_week']).mean()['sales']#.reset_index()
data_grouped_month = train.groupby(['month']).mean()['sales']
data_grouped_year = train.groupby(['year']).mean()['sales']

fig = make_subplots(rows=1, cols=3, subplot_titles=('Sales - Day of Week', 'Sales - Month', 'Sales - Year'))

# Day of Week subplot
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig.add_trace(go.Bar(x=day_names, y=list(data_grouped_day.values), marker=dict(color=list(data_grouped_day.values), colorscale='Viridis')), row=1, col=1)
fig.update_xaxes(title_text='Day of Week', row=1, col=1, tickangle=-90)
fig.update_yaxes(title_text='Mean Sales', row=1, col=1, title_standoff=1)

# Month subplot
month_names = [calendar.month_name[i] for i in range(1, 13)]
fig.add_trace(go.Bar(x=month_names, y=list(data_grouped_month.values), marker=dict(color=list(data_grouped_month.values), colorscale='Viridis')), row=1, col=2)
fig.update_xaxes(title_text='Month', row=1, col=2, tickangle=-90)
fig.update_yaxes(title_text='Mean Sales', row=1, col=2, title_standoff=1)

# Year subplot
fig.add_trace(go.Bar(x=list(data_grouped_year.index), y=list(data_grouped_year.values), marker=dict(color=list(data_grouped_year.values), colorscale='Viridis')), row=1, col=3)
fig.update_xaxes(title_text='Year', row=1, col=3, tickangle=-90, title_standoff=48, tickvals=list(data_grouped_year.index), ticktext=list(data_grouped_year.index))
fig.update_yaxes(title_text='Mean Sales', row=1, col=3, title_standoff=1)

# Update subplot layout
fig.update_layout(height=400, width=1000, showlegend=False)

# Show the plot
#fig.show(renderer='vscode')
#fig.write_image('/Users/agryah/Documents/TUAI/abc.png')
fig.write_image("/Users/agryah/Documents/TUAI/images/sales_vs_daymonthweek.png", scale=3)


# ************ store wise sales over all years ************

# Aggregate the sales data
a = train.set_index("date").groupby(["store_nbr"]).resample("D").sales.sum().reset_index()
# Create traces for each store
fig = go.Figure()
#traces = []


for contestant, group in a.groupby("store_nbr"):
    fig.add_trace(go.Scatter(x=list(group["date"]), y=list(group["sales"].values),name=contestant,
                             hovertemplate="Store=%s<br>Date=%%{x}<br>sales=%%{y}<extra></extra>"% contestant))
'''fig.update_layout(legend_title_text = "Contestant")
fig.update_xaxes(title_text="date")
fig.update_yaxes(title_text="sales")'''
fig.update_layout(
    title = dict(
        text="Daily Total Sales of the Stores",
        x=0.5,  # Centers the title
        xanchor="center"  # Ensures proper centering
    ),
    #title="Daily Total Sales of the Stores",
    xaxis_title="Date",
    yaxis_title="Sales",
    height=400,width=1000, showlegend=False
    #legend=dict(
    #    title="Contestant",  # Rename legend title
    #    orientation="h",  # Make the legend horizontal
    #    yanchor="auto", 
    #    y=-0.7,  # Move it below the chart
    #    xanchor="auto", 
    #    x=0.5,
    #),
    #margin=dict(l=40, r=40, t=40, b=100)  # Add space for the legend
)
#fig.show(renderer='vscode',scale=6)
fig.write_image("/Users/agryah/Documents/TUAI/images/storewise_sales_vs_year.png", scale=3)

# ************ Average sales by date for all stores and products ************

train_aux = train[['date', 'sales', 'onpromotion']].groupby('date').mean()
train_aux = train_aux.reset_index()
fig = go.Figure(data=go.Scatter(x=train_aux['date'].tolist(), 
                                y=train_aux['sales'].tolist(),
                                marker_color='red', text="sales"))
fig.update_layout(title=dict(
        text="Avg Sales by date for all stores and products",
        x=0.5,  # Centers the title
        xanchor="center"  # Ensures proper centering
        ),
        xaxis=dict(
        title="Date",
        title_font=dict(size=16),  # X-axis title font size
        tickfont=dict(size=12),  # X-axis tick labels font size
    ),
    yaxis=dict(
        title="Sales",
        title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12),  # Y-axis tick labels font size
    ),height=500,width=1300,showlegend=False)
fig.write_image("/Users/agryah/Documents/TUAI/images/avgsales_vs_date.png", scale=3)


# ************ Average sales by product type ************

# Group by 'family' and calculate mean sales
a = train.groupby("family").sales.mean().sort_values(ascending=False).reset_index()

# Create the bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=a["family"].tolist(), 
    x=a["sales"].tolist(), 
    orientation="h",  # Horizontal bar chart
    marker=dict(color=a["sales"].tolist(), colorscale="Viridis"),  # Color based on sales values
))

# Update layout
fig.update_layout(
    title=dict(
        text="Which Product Family is Preferred More?",
        font=dict(size=20),  # Title font size
        x=0.5,  # Centers the title
        xanchor="center"  # Ensures proper centering
    ),
    xaxis=dict(
        title="Average Sales",
        title_font=dict(size=16),  # X-axis title font size
        tickfont=dict(size=12),  # X-axis tick labels font size
    ),
    yaxis=dict(
        title="Product Family",
        tickvals=a["family"].tolist(),
        title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12),  # Y-axis tick labels font size
        categoryorder="total ascending",
    ),
    #yaxis=dict(categoryorder="total ascending"),  # Sorts categories by sales
    height=500,  # Adjust height to accommodate long labels
)


# Show figure
fig.write_image("/Users/agryah/Documents/TUAI/images/avgsales_vs_product_type.png", scale=3)

# ************ Average sales by store number ************
train_eda = train.copy()
temp = train_eda.groupby("store_nbr")["sales"].mean().sort_values(ascending=False).to_frame()

# Convert index to string for x-axis labels
store_numbers = temp.index.astype(str)
sales_values = temp["sales"].tolist()

# Create the bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=store_numbers,  
    y=sales_values, 
    marker=dict(color=sales_values, colorscale="Viridis"),  # Color gradient
))

# Ensure all store numbers are displayed on the x-axis
fig.update_layout(
    title=dict(
        text="Avg. Sales by Store No.",
        font=dict(size=20),  # Title font size
        x=0.5,  # Centers the title
        xanchor="center"  # Ensures proper centering
    ),
        
    xaxis=dict(
        title="Store No.",
        tickmode="linear",  # Ensure all ticks are shown
        tickvals=store_numbers,  # Display all store numbers
        tickangle=90,  # Rotate labels for better visibility
        title_font=dict(size=16),  # X-axis title font size
        tickfont=dict(size=12),  # X-axis tick labels font size
    ),
    yaxis=dict(
        title="Avg. Sales",
        title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12),  # Y-axis tick labels font size
    ),
    height=500
)


fig.write_image("/Users/agryah/Documents/TUAI/images/avgsales_vs_store_nbr.png", scale=3)








'''train_aux = train[train['onpromotion']>0]
fig = go.Figure(data=go.Scatter(x=train_aux['onpromotion'].tolist(), 
                                y=train_aux['sales'].tolist(),
                                marker_color='red', text="sales"))
fig.update_layout(title=dict(
        text="Avg Sales by date for all stores and products",
        x=0.5,  # Centers the title
        xanchor="center"  # Ensures proper centering
        ),xaxis_title="Date",yaxis_title="Sales",showlegend=False)
fig.write_image("/Users/agryah/Documents/TUAI/images/onpromotion_vs_sales.png", scale=3)
'''
################################################################
############### OIL ##################
oil['date'] = pd.to_datetime(oil['date'])

# Handling Missing Data in Oil Prices
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()

# Replace zero values with NaN for interpolation
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])

# Create a new column for interpolated values
oil["dcoilwtico_interpolated"] = oil["dcoilwtico"]
oil["dcoilwtico_interpolated"].interpolate(limit_direction="both", inplace=True)

# Prepare Data for Plotting
fig = go.Figure()

# Interpolated Oil Prices
fig.add_trace(go.Scatter(
    x=oil["date"].tolist(), 
    y=oil["dcoilwtico_interpolated"].tolist(), 
    mode="lines", 
    name="Interpolated Oil Price",
    line=dict(color="blue"),  # Solid blue line for interpolated data
    hoverinfo="x+y"
))

# Original Oil Prices
fig.add_trace(go.Scatter(
    x=oil["date"].tolist(), 
    y=oil["dcoilwtico"].tolist(), 
    mode="lines", 
    name="Original Oil Price",
    line=dict(color="red"),  # Dashed red line for original
    hoverinfo="x+y"
))



# Layout Settings
# fig.update_layout(
#     title=dict(
#         text="Daily Oil Price",
#         x=0.5,  # Centers the title
#         xanchor="center"  # Ensures proper centering
#     ),
#     xaxis_title="Date",
#     yaxis_title="Oil Price (USD)",height=500,width=1000,
#     legend_title="Legend",
#     template="plotly_white"
# )
fig.update_layout(
    title=dict(
        text="Daily Oil Price",
        font=dict(size=20),
        x=0.5,  # Centers the title
        xanchor="center"
    ),
    xaxis=dict(
        title="Date",
        title_font=dict(size=16),  # X-axis title font size
        tickfont=dict(size=12),  # X-axis tick labels font size
    ),
    yaxis=dict(
        title="Oil Price (USD)",
        title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12),  # Y-axis tick labels font size
    ),
    legend=dict(
        title="Legend",
        x=1,  # Position at the right
        y=1,  # Position at the top
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent background
        bordercolor="black",
        borderwidth=1
    ),
    #template="plotly_white"
    width=700
)

# Show Plot
fig.write_image("/Users/agryah/Documents/TUAI/images/daily_oil_price.png", scale=3)


################################################################
############### Transactions ##################
transactions = transactions.sort_values(["store_nbr","date"])
transactions['date'] = pd.to_datetime(transactions['date'])
fig = make_subplots(rows=1, cols=1, subplot_titles=("Transactions by Month",))

# Define colors for months
colors_month = [
    'red', 'green', 'blue',
    'magenta', 'yellow', 'brown',
    'pink', 'darkgrey', 'orange',
    'darkblue', 'olive', 'purple'
]

years = transactions['date'].dt.year.unique()
for year in years:
    trace_data = []
    for month in range(1, 13):
        data_month = transactions[(transactions['date'].dt.year == year) & (transactions['date'].dt.month == month)]
        trace = go.Box(
            x=[f"{calendar.month_abbr[month]}-{year}"] * len(data_month['transactions']), 
            y=data_month['transactions'].tolist(),
            boxpoints='outliers', 
            jitter=0.4, 
            pointpos=0,
            marker=dict(color=colors_month[month - 1]),
            name=f"{calendar.month_abbr[month]}-{year}"
        )
        trace_data.append(trace)

    fig.add_traces(trace_data, rows=[1] * len(trace_data), cols=[1] * len(trace_data))

# Update X and Y axes
fig.update_xaxes(
    title_text="Month-Year", 
    tickvals=[f"{calendar.month_abbr[month]}-{year}" for year in years for month in range(1, 13)],
    ticktext=[f"{calendar.month_abbr[month]}-{year}" for year in years for month in range(1, 13)], 
    row=1, col=1
)
fig.update_yaxes(title_text="Transactions", row=1, col=1)

# Update layout
fig.update_layout(xaxis=dict(
        title_font=dict(size=16),  # X-axis title font size
        tickfont=dict(size=12),  # X-axis tick labels font size
    ),
    yaxis=dict(
        title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12),  # Y-axis tick labels font size
    ),
    height=600, width=800, 
    showlegend=False
)

# Show the plot
fig.write_image("/Users/agryah/Documents/TUAI/images/transactions_by_month.png", scale=3)

################################################################
############### Did Earthquake affect the sale ##################
train['year_month'] = train['date'].dt.to_period('M').astype(str)

# group by year-month and calculate average sales
avg_sales = train.groupby('year_month')['sales'].mean()
fig = go.Figure()

# Plot the average sales
fig.add_trace(go.Scatter(x=avg_sales.index, y=avg_sales.values.tolist(), mode='lines', name='Average Sales'))

earthquake_date = '2016-04-16'

# Add a vertical line for the earthquake in April 2016
fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=earthquake_date,
        x1=earthquake_date,
        y0=avg_sales.values.min()-50,
        y1=avg_sales.values.max()+50,
        line=dict(color="red", dash="dash")
    )
)
fig.add_annotation(
    x=earthquake_date,
    y=avg_sales.values.max(),  # Positioning at the top
    text="16 Apr-2016 Earthquake",
    showarrow=True,
    arrowhead=2,
    ax=40,  # Adjusting text position
    ay=-40,
    bgcolor="white"
)

# Calculate the start and end dates for the 10-day range
start_date = '2012-11-1'
end_date = '2017-8-30'


fig.update_layout(
    xaxis=dict(title='Year-Month', range=[start_date, end_date],title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12)),
    yaxis=dict(title='Sales',title_font=dict(size=16),  # Y-axis title font size
        tickfont=dict(size=12)),
    title=dict(
        text="Average Sales Over Time",
        x=0.5,  # Centers the title
        xanchor="center"  # Ensures proper centering
    ),
    showlegend=False,
    height=400,
    margin=dict(l=5, r=10, t=40, b=30)
)
fig.write_image("/Users/agryah/Documents/TUAI/images/avgsales_with_earthquake.png", scale=3)
