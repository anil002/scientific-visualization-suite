import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import joypy  # For Ridgeline Plot
from scipy.cluster import hierarchy  # For Dendrogram
from io import BytesIO
import base64
from wordcloud import WordCloud  # For Word Cloud

# Set page config
st.set_page_config(page_title="Advanced Scientific Visualization Suite", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("🔬 Advanced Scientific Visualization Suite")
st.markdown("""
    Welcome to the **Advanced Scientific Visualization Suite**. Upload your Excel/CSV file to explore a wide range of professional-grade scientific visualizations, including the latest trends as of 2025. Customize your plots with advanced options and export them for publication or analysis.
""")

# Sidebar for file upload and data options
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'], help="Supported formats: CSV, Excel (.xlsx)")

# Generate dummy data if no file is uploaded
if uploaded_file is None:
    st.sidebar.info("No file uploaded. Using synthetic dataset.")
    np.random.seed(42)
    data = {
        'Time': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Temperature (°C)': np.random.normal(25, 5, 100),
        'Precipitation (mm)': np.random.exponential(10, 100),
        'Flow_Rate (m³/s)': np.random.uniform(0, 100, 100),
        'Pressure (hPa)': np.random.normal(1000, 50, 100),
        'pH': np.random.uniform(6, 8, 100),
        'Salinity (ppt)': np.random.normal(35, 2, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Wind_Speed (m/s)': np.random.uniform(0, 20, 100),
        'Wind_Direction (°)': np.random.uniform(0, 360, 100),
        'Velocity_X (m/s)': np.random.normal(0, 5, 100),
        'Velocity_Y (m/s)': np.random.normal(0, 5, 100),
        'Text_Data': np.random.choice(['climate', 'weather', 'rain', 'temperature', 'pressure'], 100)  # For Word Cloud
    }
    df = pd.DataFrame(data)
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Data filtering
st.sidebar.header("Data Filters")
filter_col = st.sidebar.selectbox("Filter by column", ["None"] + df.columns.tolist(), index=0)
if filter_col != "None":
    filter_values = st.sidebar.multiselect(f"Select {filter_col} values", df[filter_col].unique())
    if filter_values:
        df = df[df[filter_col].isin(filter_values)]

# Display data preview
st.subheader("Data Preview")
st.write(df.head())
st.write(f"Dataset Shape: {df.shape}")

# Dictionary of plot definitions and uses
plot_info = {
    "Scatter Plot": {
        "Definition": "A plot displaying individual data points on a two-dimensional plane, with each axis representing a variable.",
        "Uses": "Identify relationships, trends, or clusters between two continuous variables (e.g., temperature vs. precipitation)."
    },
    "Histogram": {
        "Definition": "A bar plot showing the frequency distribution of a single variable by dividing data into bins.",
        "Uses": "Analyze the distribution of a numeric variable (e.g., frequency of pH values)."
    },
    "Bar Chart": {
        "Definition": "A plot using rectangular bars to represent categorical data, with bar height indicating values.",
        "Uses": "Compare quantities across categories (e.g., average flow rate by category)."
    },
    "Line Plot": {
        "Definition": "A plot connecting data points with lines, typically used for continuous data over time.",
        "Uses": "Visualize trends over time or another continuous variable (e.g., temperature over days)."
    },
    "Box Plot": {
        "Definition": "A plot summarizing data distribution through quartiles, showing median, spread, and outliers.",
        "Uses": "Compare distributions across categories (e.g., salinity by category)."
    },
    "Violin Plot": {
        "Definition": "A combination of box plot and kernel density plot, showing data distribution and density.",
        "Uses": "Examine distribution shapes across groups (e.g., pressure distribution by category)."
    },
    "Heatmap": {
        "Definition": "A matrix plot where colors represent values, often showing correlations.",
        "Uses": "Visualize relationships between multiple variables (e.g., correlation matrix of environmental factors)."
    },
    "Contour Plot": {
        "Definition": "A plot showing 3D data in 2D using contour lines to represent levels of a third variable.",
        "Uses": "Map relationships between three variables (e.g., temperature, pressure, and salinity)."
    },
    "Pair Plot": {
        "Definition": "A grid of scatter plots showing pairwise relationships between multiple variables.",
        "Uses": "Explore correlations and distributions in multivariate data (e.g., all numeric environmental variables)."
    },
    "Area Plot": {
        "Definition": "A plot filling the area under a line, often used for cumulative data.",
        "Uses": "Show cumulative trends over time (e.g., total precipitation over months)."
    },
    "Pie Chart": {
        "Definition": "A circular chart divided into slices representing proportions of a categorical variable.",
        "Uses": "Display percentage breakdowns (e.g., proportion of categories in a dataset)."
    },
    "3D Surface Plot": {
        "Definition": "A 3D plot showing a surface where height represents a third variable.",
        "Uses": "Visualize complex relationships in three dimensions (e.g., flow rate over time and pressure)."
    },
    "Taylor Diagram": {
        "Definition": "A polar plot comparing model performance using standard deviation and correlation.",
        "Uses": "Evaluate model accuracy against reference data (e.g., modeled vs. observed temperature)."
    },
    "Polar Plot": {
        "Definition": "A plot on a circular grid, often used for directional or cyclic data.",
        "Uses": "Display directional data (e.g., wind speed vs. direction)."
    },
    "Bubble Plot": {
        "Definition": "A scatter plot where point size represents a third variable.",
        "Uses": "Show three variables simultaneously (e.g., temperature, precipitation, and flow rate)."
    },
    "Density Plot": {
        "Definition": "A smoothed plot showing the distribution of data, often in 2D.",
        "Uses": "Visualize density of two continuous variables (e.g., temperature vs. pressure)."
    },
    "Time Series Plot": {
        "Definition": "A line plot specifically for data over time.",
        "Uses": "Track changes over time (e.g., daily salinity measurements)."
    },
    "Q-Q Plot": {
        "Definition": "A plot comparing two probability distributions by plotting their quantiles.",
        "Uses": "Assess normality or compare distributions (e.g., check if pH is normally distributed)."
    },
    "Ridgeline Plot": {
        "Definition": "Overlapping density plots for multiple groups, stacked vertically.",
        "Uses": "Compare distributions across categories (e.g., temperature by category)."
    },
    "Parallel Coordinates Plot": {
        "Definition": "A plot showing multiple variables as parallel axes, with lines connecting data points.",
        "Uses": "Analyze multivariate data relationships (e.g., all numeric variables in a dataset)."
    },
    "Sankey Diagram": {
        "Definition": "A flow diagram showing the magnitude of flow between nodes.",
        "Uses": "Visualize energy, material, or data flows (e.g., source to destination flows)."
    },
    "Correlogram": {
        "Definition": "A pair plot with scatter plots and histograms to show variable relationships.",
        "Uses": "Explore correlations in multivariate data (e.g., environmental measurements)."
    },
    "Wind Rose": {
        "Definition": "A circular plot showing wind speed and direction distribution.",
        "Uses": "Analyze wind patterns (e.g., wind speed by direction)."
    },
    "Streamplot": {
        "Definition": "A plot showing flow lines of a vector field.",
        "Uses": "Visualize fluid dynamics or vector fields (e.g., velocity fields)."
    },
    "Dendrogram": {
        "Definition": "A tree-like plot showing hierarchical clustering.",
        "Uses": "Identify clusters in data (e.g., grouping environmental samples)."
    },
    "Error Bar Plot": {
        "Definition": "A plot with bars showing means and error bars for uncertainty.",
        "Uses": "Compare means with variability (e.g., average temperature with standard deviation)."
    },
    "Boxen Plot": {
        "Definition": "An enhanced box plot showing more quantiles for large datasets.",
        "Uses": "Analyze detailed distributions (e.g., precipitation across categories)."
    },
    "Lag Plot": {
        "Definition": "A scatter plot of a variable against its lagged values.",
        "Uses": "Check for autocorrelation in time series (e.g., temperature lag analysis)."
    },
    "Autocorrelation Plot": {
        "Definition": "A plot showing correlation of a time series with its own lagged values.",
        "Uses": "Detect periodicity or trends (e.g., pressure autocorrelation)."
    },
    "Interactive 3D Scatter": {
        "Definition": "An interactive 3D scatter plot for exploring three variables.",
        "Uses": "Investigate complex 3D relationships (e.g., temperature, pressure, and salinity)."
    },
    "Hexbin Plot": {
        "Definition": "A 2D histogram using hexagons to show density of overlapping points.",
        "Uses": "Visualize high-density scatter data (e.g., temperature vs. precipitation)."
    },
    "Sunburst Chart": {
        "Definition": "A hierarchical pie chart showing nested categories.",
        "Uses": "Display hierarchical data (e.g., category breakdowns with subcategories)."
    },
    "Animated Time Series": {
        "Definition": "A dynamic line plot showing changes over time with animation.",
        "Uses": "Visualize temporal evolution (e.g., temperature changes over days)."
    },
    "Word Cloud": {
        "Definition": "A visual representation of text data where word size reflects frequency or importance.",
        "Uses": "Summarize text data (e.g., common terms in environmental notes)."
    },
    "Facet Grid Plot": {
        "Definition": "A multi-panel plot showing subsets of data across categories.",
        "Uses": "Compare distributions across groups (e.g., temperature by category)."
    }
}

# Visualization options
st.sidebar.header("Visualization Settings")
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    list(plot_info.keys()),
    help="Choose from a variety of scientific visualizations, including 2025 trends."
)

# Display plot definition and uses
st.sidebar.subheader("About This Plot")
st.sidebar.markdown(f"**Definition**: {plot_info[plot_type]['Definition']}")
st.sidebar.markdown(f"**Uses**: {plot_info[plot_type]['Uses']}")

# Dynamic column selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

x_col = st.sidebar.selectbox("X-axis column", all_cols, help="Select the column for the X-axis.")
y_col = st.sidebar.selectbox("Y-axis column", numeric_cols, index=0 if numeric_cols else None) if plot_type not in ["Histogram", "Pie Chart", "Pair Plot", "Ridgeline Plot", "Correlogram", "Dendrogram", "Lag Plot", "Autocorrelation Plot", "Word Cloud"] else None
z_col = st.sidebar.selectbox("Z-axis column (if applicable)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0) if plot_type in ["3D Surface Plot", "Bubble Plot", "Contour Plot", "Interactive 3D Scatter"] else None
hue_col = st.sidebar.selectbox("Hue column (optional)", ["None"] + categorical_cols, index=0) if plot_type in ["Scatter Plot", "Box Plot", "Violin Plot", "Bar Chart", "Error Bar Plot", "Boxen Plot", "Facet Grid Plot"] else None

# Plot customization
st.sidebar.subheader("Customization")
color = st.sidebar.color_picker("Primary color", "#00f900")
log_scale_x = st.sidebar.checkbox("Log scale X-axis", value=False) if plot_type in ["Scatter Plot", "Line Plot", "Histogram", "Density Plot", "Hexbin Plot"] else None
log_scale_y = st.sidebar.checkbox("Log scale Y-axis", value=False) if plot_type in ["Scatter Plot", "Line Plot", "Bar Chart", "Area Plot", "Density Plot", "Hexbin Plot"] else None
fig_width = st.sidebar.slider("Figure width", 4, 12, 8)
fig_height = st.sidebar.slider("Figure height", 4, 12, 6)

# Function for Taylor Diagram
def taylor_diagram(ref, model):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw={'projection': 'polar'})
    ref_std = np.std(ref)
    model_std = np.std(model)
    correlation = np.corrcoef(ref, model)[0,1]
    ax.plot(0, ref_std, 'ko', label='Reference')
    theta = np.arccos(correlation)
    r = model_std
    ax.plot(theta, r, 'r*', label='Model')
    ax.set_rmax(1.5 * max(ref_std, model_std))
    ax.grid(True)
    ax.set_title("Taylor Diagram")
    ax.legend()
    return fig

# Generate visualization
st.subheader(f"{plot_type}")
try:
    if plot_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col if hue_col != "None" else None, color_discrete_sequence=[color], log_x=log_scale_x, log_y=log_scale_y)
        fig.update_layout(width=fig_width*100, height=fig_height*100)
        st.plotly_chart(fig)

    elif plot_type == "Histogram":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.histplot(data=df, x=x_col, color=color, log_scale=log_scale_x, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if hue_col != "None":
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=[color], ax=ax)
        else:
            df.groupby(x_col)[y_col].mean().plot(kind='bar', color=color, ax=ax)
        ax.set_ylabel(y_col)
        if log_scale_y:
            ax.set_yscale('log')
        st.pyplot(fig)

    elif plot_type == "Line Plot":
        fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=[color], log_x=log_scale_x, log_y=log_scale_y)
        fig.update_layout(width=fig_width*100, height=fig_height*100)
        st.plotly_chart(fig)

    elif plot_type == "Box Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None, color=color, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Violin Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None, color=color, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Heatmap":
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.error("Heatmap requires at least two numeric columns")

    elif plot_type == "Contour Plot":
        if len(numeric_cols) >= 3:
            x, y, z = df[x_col], df[y_col], df[z_col]
            X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 20), np.linspace(y.min(), y.max(), 20))
            Z = pd.DataFrame({x_col: X.ravel(), y_col: Y.ravel()}).apply(lambda row: z.iloc[np.argmin((x - row[x_col])**2 + (y - row[y_col])**2)], axis=1).values.reshape(20, 20)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            contour = ax.contourf(X, Y, Z, cmap="viridis")
            plt.colorbar(contour, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Contour Plot requires at least three numeric columns")

    elif plot_type == "Pair Plot":
        if len(numeric_cols) >= 2:
            fig = sns.pairplot(df[numeric_cols], height=fig_height/2)
            st.pyplot(fig)
        else:
            st.error("Pair Plot requires at least two numeric columns")

    elif plot_type == "Area Plot":
        fig = px.area(df, x=x_col, y=y_col, color_discrete_sequence=[color], log_y=log_scale_y)
        fig.update_layout(width=fig_width*100, height=fig_height*100)
        st.plotly_chart(fig)

    elif plot_type == "Pie Chart":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        df[x_col].value_counts().plot(kind='pie', colors=[color, "#ff9999", "#66b3ff"], ax=ax)
        st.pyplot(fig)

    elif plot_type == "3D Surface Plot":
        if len(numeric_cols) >= 3:
            x, y, z = df[x_col], df[y_col], df[z_col]
            fig = go.Figure(data=[go.Surface(z=z.values.reshape(10, 10), x=x[:100:10], y=y[:100:10])])
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("3D Surface Plot requires at least three numeric columns")

    elif plot_type == "Taylor Diagram":
        if len(numeric_cols) >= 2:
            ref_data = df[numeric_cols[0]]
            model_data = df[numeric_cols[1]]
            fig = taylor_diagram(ref_data, model_data)
            st.pyplot(fig)
        else:
            st.error("Taylor Diagram requires at least two numeric columns")

    elif plot_type == "Polar Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw={'projection': 'polar'})
        ax.plot(df[x_col], df[y_col], color=color)
        st.pyplot(fig)

    elif plot_type == "Bubble Plot":
        if len(numeric_cols) >= 3:
            fig = px.scatter(df, x=x_col, y=y_col, size=z_col, color_discrete_sequence=[color])
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("Bubble Plot requires at least three numeric columns")

    elif plot_type == "Density Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.kdeplot(data=df, x=x_col, y=y_col, fill=True, cmap="Blues", ax=ax)
        if log_scale_x:
            ax.set_xscale('log')
        if log_scale_y:
            ax.set_yscale('log')
        st.pyplot(fig)

    elif plot_type == "Time Series Plot":
        fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        fig.update_layout(width=fig_width*100, height=fig_height*100)
        st.plotly_chart(fig)

    elif plot_type == "Q-Q Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        stats.probplot(df[x_col].dropna(), dist="norm", plot=ax)
        ax.set_title("Q-Q Plot")
        st.pyplot(fig)

    elif plot_type == "Ridgeline Plot":
        if len(numeric_cols) >= 1 and 'Category' in df.columns:
            fig, axes = joypy.joyplot(df, by="Category", column=x_col, colormap=plt.cm.Blues, fade=True, figsize=(fig_width, fig_height))
            st.pyplot(fig)
        else:
            st.error("Ridgeline Plot requires a numeric column and 'Category' column")

    elif plot_type == "Parallel Coordinates Plot":
        if len(numeric_cols) >= 2:
            fig = px.parallel_coordinates(df, dimensions=numeric_cols, color_continuous_scale=px.colors.diverging.Tealrose)
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("Parallel Coordinates Plot requires at least two numeric columns")

    elif plot_type == "Sankey Diagram":
        if 'Source' in df.columns and 'Destination' in df.columns and 'Value' in df.columns:
            labels = list(set(df['Source'].tolist() + df['Destination'].tolist()))
            source_indices = [labels.index(s) for s in df['Source']]
            target_indices = [labels.index(t) for t in df['Destination']]
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=labels),
                link=dict(source=source_indices, target=target_indices, value=df['Value'])
            )])
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("Sankey Diagram requires 'Source', 'Destination', and 'Value' columns")

    elif plot_type == "Correlogram":
        if len(numeric_cols) >= 2:
            fig = sns.pairplot(df[numeric_cols], kind="scatter", diag_kind="hist", plot_kws={'color': color}, height=fig_height/2)
            st.pyplot(fig)
        else:
            st.error("Correlogram requires at least two numeric columns")

    elif plot_type == "Wind Rose":
        if 'Wind_Speed' in df.columns and 'Wind_Direction' in df.columns:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw={'projection': 'polar'})
            ax.bar(np.radians(df['Wind_Direction']), df['Wind_Speed'], width=0.5, color=color, edgecolor='black')
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.pi/2)
            ax.set_title("Wind Rose")
            st.pyplot(fig)
        else:
            st.error("Wind Rose requires 'Wind_Speed' and 'Wind_Direction' columns")

    elif plot_type == "Streamplot":
        if 'Velocity_X' in df.columns and 'Velocity_Y' in df.columns:
            x = np.linspace(df[x_col].min(), df[x_col].max(), 20)
            y = np.linspace(df[y_col].min(), df[y_col].max(), 20)
            X, Y = np.meshgrid(x, y)
            U = df['Velocity_X'].values[:400].reshape(20, 20) if len(df) >= 400 else np.tile(df['Velocity_X'].values, (20, 20))[:20, :20]
            V = df['Velocity_Y'].values[:400].reshape(20, 20) if len(df) >= 400 else np.tile(df['Velocity_Y'].values, (20, 20))[:20, :20]
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.streamplot(X, Y, U, V, color=color)
            ax.set_title("Streamplot")
            st.pyplot(fig)
        else:
            st.error("Streamplot requires 'Velocity_X' and 'Velocity_Y' columns for vector field")

    elif plot_type == "Dendrogram":
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            Z = hierarchy.linkage(df[numeric_cols].dropna(), method='ward')
            hierarchy.dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color=color)
            ax.set_title("Dendrogram")
            st.pyplot(fig)
        else:
            st.error("Dendrogram requires at least two numeric columns")

    elif plot_type == "Error Bar Plot":
        if len(numeric_cols) >= 1 and x_col and y_col:
            grouped = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.errorbar(grouped[x_col], grouped['mean'], yerr=grouped['std'], fmt='o-', color=color, capsize=5)
            ax.set_title("Error Bar Plot")
            ax.set_ylabel(y_col)
            st.pyplot(fig)
        else:
            st.error("Error Bar Plot requires at least two columns")

    elif plot_type == "Boxen Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxenplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None, color=color, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Lag Plot":
        if x_col in numeric_cols:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            pd.plotting.lag_plot(df[x_col], lag=1, ax=ax, c=color)
            ax.set_title("Lag Plot (Lag=1)")
            st.pyplot(fig)
        else:
            st.error("Lag Plot requires a numeric column")

    elif plot_type == "Autocorrelation Plot":
        if x_col in numeric_cols:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            pd.plotting.autocorrelation_plot(df[x_col].dropna(), ax=ax, color=color)
            ax.set_title("Autocorrelation Plot")
            st.pyplot(fig)
        else:
            st.error("Autocorrelation Plot requires a numeric column")

    elif plot_type == "Interactive 3D Scatter":
        if len(numeric_cols) >= 3:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=hue_col if hue_col != "None" else None, color_discrete_sequence=[color])
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("Interactive 3D Scatter requires at least three numeric columns")

    elif plot_type == "Hexbin Plot":
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        hb = ax.hexbin(df[x_col], df[y_col], gridsize=30, cmap="Blues", mincnt=1, bins='log' if log_scale_x or log_scale_y else None)
        plt.colorbar(hb, ax=ax)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        if log_scale_x:
            ax.set_xscale('log')
        if log_scale_y:
            ax.set_yscale('log')
        st.pyplot(fig)

    elif plot_type == "Sunburst Chart":
        if len(categorical_cols) >= 2:
            fig = px.sunburst(df, path=categorical_cols[:2], values=y_col if y_col else None, color_discrete_sequence=[color])
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("Sunburst Chart requires at least two categorical columns")

    elif plot_type == "Animated Time Series":
        if 'Time' in df.columns and y_col:
            fig = px.line(df, x='Time', y=y_col, animation_frame='Time', color_discrete_sequence=[color])
            fig.update_layout(width=fig_width*100, height=fig_height*100)
            st.plotly_chart(fig)
        else:
            st.error("Animated Time Series requires a 'Time' column and a numeric Y-axis column")

    elif plot_type == "Word Cloud":
        if 'Text_Data' in df.columns or x_col in categorical_cols:
            text = ' '.join(df[x_col].astype(str))
            wordcloud = WordCloud(width=fig_width*100, height=fig_height*100, background_color='white', colormap='Blues').generate(text)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.error("Word Cloud requires a text or categorical column")

    elif plot_type == "Facet Grid Plot":
        if len(numeric_cols) >= 1 and hue_col != "None":
            g = sns.catplot(data=df, x=x_col, y=y_col, col=hue_col, kind="box", height=fig_height, aspect=fig_width/fig_height, color=color)
            fig = g.figure
            st.pyplot(fig)
        else:
            st.error("Facet Grid Plot requires a numeric column and a hue column")

    # Export options
    if 'fig' in locals():
        buf = BytesIO()
        if isinstance(fig, go.Figure):  # Plotly figure
            fig.write_image(buf, format="png", width=fig_width*100, height=fig_height*100)
        elif isinstance(fig, plt.Figure):  # Matplotlib figure
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        else:  # Seaborn pairplot, joyplot, or facet grid
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            plt.close()  # Close the figure to free memory
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{plot_type.lower().replace(" ", "_")}.png">Download Plot as PNG</a>'
        st.markdown(href, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error generating plot: {str(e)}")

# Add download button for data
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)
st.sidebar.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv',
    help="Download the current filtered dataset."
)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Dr.Anil Kumar Singh| Powered by Streamlit | © 2025")