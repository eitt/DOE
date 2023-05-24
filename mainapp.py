import streamlit as st
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations


# Define levels for each factor
levels = {
    'Temperature': ['low', 'medium', 'high'],
    'Pressure': ['low', 'medium', 'high'],
    'Thinner':  ['low', 'medium', 'high']
}

# Default factor names
factor_names = {
    "Temperature": "Temperature",
    "Pressure": "Pressure",
    "Thinner": "Thinner"
}


def create_dataframe(replications=2):
    # Generate all combinations of factor levels
    combinations = list(itertools.product(*levels.values()))

    # Repeat the combinations for the number of replications
    combinations = combinations * replications

    # Create a DataFrame with the factorial design
    df = pd.DataFrame(combinations, columns=levels.keys())

    # Map factor levels to numeric values
    mappings = {'low': -1, 'medium': 0, 'high': 1}
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(mappings)

    # Rename columns with custom factor names
    # Renaming dataframe columns
    # Rename the columns in df
    df.rename(columns={old: new for old, new in factor_names.items()}, inplace=True)


    return df
def add_numeric_columns(df, factor_names):
    """
    Create numeric columns for each factor in the dataframe.
    """

    # Numeric columns for factors
    for factor, new_name in factor_names.items():
        df[f'{new_name}_num'] = df[new_name].map(factor_values[factor])

    return df


def compute_y(df, coefficients, factor_names):
    noise = np.random.normal(0, 2, len(df))  # 0 is the mean of the normal distribution, and 1 is the standard deviation
    df['Y'] = (coefficients[0]
               + coefficients[1]*df[f'{factor_names["Temperature"]}_num']
               + coefficients[2]*df[f'{factor_names["Pressure"]}_num']
               + coefficients[3]*df[f'{factor_names["Thinner"]}_num']
               + coefficients[4]*df[f'{factor_names["Temperature"]}_num']*df[f'{factor_names["Pressure"]}_num']
               + coefficients[5]*df[f'{factor_names["Temperature"]}_num']*df[f'{factor_names["Thinner"]}_num']
               + coefficients[6]*df[f'{factor_names["Pressure"]}_num']*df[f'{factor_names["Thinner"]}_num']
               + noise)
    return df



def plot_3d(df):
    fig = go.Figure(data=go.Scatter3d(
        x=df[f'{factor_names["Temperature"]}_num'], 
        y=df[f'{factor_names["Pressure"]}_num'], 
        z=df[f'{factor_names["Thinner"]}_num'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['Y'], 
            colorbar=dict(title='Y'), 
            opacity=0.8
        ))
    )

    fig.update_layout(scene = dict(
        xaxis_title=factor_names["Temperature"],
        yaxis_title=factor_names["Pressure"],
        zaxis_title=factor_names["Thinner"]
    ))
    st.plotly_chart(fig)
    

def plot_surface(df, factor1, factor2):
    # Create a pivot table for the surface plot
    pivot_table = df.pivot_table(values='Y', index=f'{factor1}_num', columns=f'{factor2}_num')

    # Create a 3D plot
    fig = go.Figure(data=[go.Surface(z=pivot_table.values, x=pivot_table.index, y=pivot_table.columns)])

    # Add labels
    fig.update_layout(scene = dict(
        xaxis_title=factor1,
        yaxis_title=factor2,
        zaxis_title='Y'
    ))
    st.plotly_chart(fig)


def plot_boxplot(df, groupby):
    fig = go.Figure()
    
    # Check if interaction term is selected
    if "*" in groupby:
        factors = groupby.split("*")
        xaxis_title = f"{factors[0]} * {factors[1]}"
        df['Interaction'] = df[f'{factors[0]}'].astype(str) + " * " + df[f'{factors[1]}'].astype(str)
        groupby = 'Interaction'
    elif groupby in factor_names.values():
        xaxis_title = groupby
        groupby = f'{groupby}_num' 
    else:
        xaxis_title = groupby

    # Group by the custom name
    groups = df.groupby(groupby)

    for name, group in groups:
        fig.add_trace(go.Box(y=group['Y'], name=str(name), boxmean=True))

    # Add labels
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Y',
    )
    st.plotly_chart(fig)


def fit_model(df, factor_names):
    formula = f"Y ~ Q('{factor_names['Temperature']}_num') + Q('{factor_names['Pressure']}_num') + Q('{factor_names['Thinner']}_num') + \
                Q('{factor_names['Temperature']}_num'):Q('{factor_names['Pressure']}_num') + \
                Q('{factor_names['Temperature']}_num'):Q('{factor_names['Thinner']}_num') + \
                Q('{factor_names['Pressure']}_num'):Q('{factor_names['Thinner']}_num')"
    model = smf.ols(formula, data=df)
    results = model.fit()
    return results
    
def print_equation(results, factor_names):
    coefs = results.params
    coef_str = [f'{coef:.2f}' if not np.isclose(coef, 1) else '' for coef in coefs]

    equation = f'Y = {coef_str[0]}'
    terms = ['Intercept', f"{factor_names['Temperature']}_num", f"{factor_names['Pressure']}_num", f"{factor_names['Thinner']}_num",
             f"{factor_names['Temperature']}_num:{factor_names['Pressure']}_num",
             f"{factor_names['Temperature']}_num:{factor_names['Thinner']}_num",
             f"{factor_names['Pressure']}_num:{factor_names['Thinner']}_num"]

    for coef, term in zip(coef_str[1:], terms[1:]):
        equation += f' + {coef}*{term}'

    return equation



def Three_factorial():
    st.title('Factorial Design Visualization')
    st.markdown("By **Leonardo H. Talero-Sarmiento** [View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    
   # Assign new names
    global factor_names
    factor_names["Temperature"] = st.sidebar.text_input("Enter a custom name for Temperature", "Temperature")
    factor_names["Pressure"] = st.sidebar.text_input("Enter a custom name for Pressure", "Pressure")
    factor_names["Thinner"] = st.sidebar.text_input("Enter a custom name for Thinner", "Thinner")



    
       
    coefficients = [
        st.sidebar.slider(f'Coefficient 0 (Intercept)', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 1 ({factor_names["Temperature"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 2 ({factor_names["Pressure"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 3 ({factor_names["Thinner"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 4 ({factor_names["Temperature"]}*{factor_names["Pressure"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 5 ({factor_names["Temperature"]}*{factor_names["Thinner"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 6 ({factor_names["Pressure"]}*{factor_names["Thinner"]})', -10.0, 10.0, 0.0),
    ]
    
    df = create_dataframe()

    # rename the columns in the DataFrame to match the user's custom names
    df.rename(columns={f"{factor}_num": f"{factor_names[factor]}_num" for factor in ["Temperature", "Pressure", "Thinner"]}, inplace=True)
    st.write(df.head())  # Print the first few rows of the DataFrame
    
    # Compute Y
    df = compute_y(df, coefficients, factor_names)
    st.subheader('Dataframe')
    st.dataframe(df)
    
    csv = df.to_csv(index=False)
    st.download_button(
        "Download data as CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv")

    st.subheader('Factors space')
    # Before plotting
    st.write(df.columns)

    plot_3d(df)
    st.subheader('Analysis of Y based on a source of variability')
    st.subheader('Box plot')
    groupby_options = list(factor_names.values()) + [f'{i}*{j}' for i, j in combinations(factor_names.values(), 2)]
    groupby = st.selectbox('Group by', groupby_options, index=0)

    if groupby == f'{factor_names["Temperature"]}*{factor_names["Pressure"]}':
        df['Interaction'] = df[f'{factor_names["Temperature"]}_num'].astype(str) + '*' + df[f'{factor_names["Pressure"]}_num'].astype(str)
    elif groupby == f'{factor_names["Temperature"]}*{factor_names["Thinner"]}':
        df['Interaction'] = df[f'{factor_names["Temperature"]}_num'].astype(str) + '*' + df[f'{factor_names["Thinner"]}_num'].astype(str)
    elif groupby == f'{factor_names["Pressure"]}*{factor_names["Thinner"]}':
        df['Interaction'] = df[f'{factor_names["Pressure"]}_num'].astype(str) + '*' + df[f'{factor_names["Thinner"]}_num'].astype(str)
    else:
        df['Interaction'] = df[f'{groupby}_num']

    plot_boxplot(df, 'Interaction')


    st.subheader('Surface plot')

    # And in the plot_surface function call
    factors = list(factor_names.values())
    factor1 = st.selectbox('Select first factor', factors, index=0)
    factor2 = st.selectbox('Select second factor', factors, index=1)
    plot_surface(df, factor1, factor2)

    st.subheader('Model fitting')

    # Similarly, when you call the fit_model and print_equation functions
    results = fit_model(df, factor_names)
    st.latex(print_equation(results, factor_names))

    st.text(results.summary())
















######################################################################
# Define levels for two-factor, two-level design
two_level_factor_values = {
    'FactorA': ['low', 'high'],
    'FactorB': ['low', 'high']
}

# Define numeric mapping for two-level design
two_level_mapping = {'low': -1, 'high': 1}

def create_two_level_dataframe(replications=3):
    # Generate all combinations of factor levels
    combinations = list(itertools.product(*two_level_factor_values.values()))

    # Repeat the combinations for the number of replications
    combinations = combinations * replications

    # Create a DataFrame with the factorial design
    df = pd.DataFrame(combinations, columns=two_level_factor_values.keys())

    # Map factor levels to numeric values
    for col in df.columns:
        df[f'{col}_num'] = df[col].map(two_level_mapping)

    return df

def compute_twolevel_y(df, coefficients, factor_names):
    noise = np.random.normal(0, 2, len(df))  
    df['Y'] = (coefficients[0]
               + coefficients[1]*df[f'{factor_names["FactorA"]}_num']
               + coefficients[2]*df[f'{factor_names["FactorB"]}_num']
               + coefficients[3]*df[f'{factor_names["FactorA"]}_num']*df[f'{factor_names["FactorB"]}_num']
               + noise)
    return df

def fit_model_twolevels(df, factor_names):
    formula = f"Y ~ Q('{factor_names['FactorA']}_num') + Q('{factor_names['FactorB']}_num') + Q('{factor_names['FactorA']}_num'):Q('{factor_names['FactorB']}_num')"
    model = smf.ols(formula, data=df)
    results = model.fit()
    return results

def plot_boxplot2D(df, factor_names):
    fig = go.Figure()

    # Loop through each factor name to create a box plot
    for factor in factor_names.values():
        fig.add_trace(go.Box(y=df[factor+'_num'], name=factor))

    # Update layout to include titles
    fig.update_layout(
        xaxis_title="Factor",
        yaxis_title="Value",
        title="Box Plot for Two-Level Factorial Design"
    )
    
    st.plotly_chart(fig)

def plot_surface_twolevels(df, factor_names):
    # Create a pivot table with the average 'Y' values for each combination of 'FactorA_num' and 'FactorB_num'
    pivot_df = df.pivot_table('Y', index=factor_names['FactorA']+'_num', columns=factor_names['FactorB']+'_num')

    # Create a surface plot
    fig = go.Figure(data=go.Surface(z=pivot_df.values,
                                    x=pivot_df.index,
                                    y=pivot_df.columns))

    fig.update_layout(title='Surface plot', autosize=False,
                      width=1000, height=500,
                      scene=dict(xaxis_title=factor_names['FactorA'],
                                 yaxis_title=factor_names['FactorB'],
                                 zaxis_title='Y'),
                      margin=dict(l=65, r=50, b=65, t=90))

    st.plotly_chart(fig)


    
def plot_2d(df, factor_names):
    fig = go.Figure(data=go.Scatter(
        x=df[f'{factor_names["FactorA"]}_num'], 
        y=df[f'{factor_names["FactorB"]}_num'], 
        mode='markers', 
        marker=dict(
            size=8,
            color=df['Y'], # set color to Y
            colorscale='Viridis', # choose a colorscale
            colorbar=dict(title='Y') # title for colorbar
        )
    ))

    fig.update_layout(
        xaxis_title=factor_names["FactorA"],
        yaxis_title=factor_names["FactorB"],
        title='2D Plot for Two-Level Factorial Design'
    )
    st.plotly_chart(fig)

def print_equation_twolevels(results, factor_names):
    coefs = results.params
    coef_str = [f'{coef:.2f}' if not np.isclose(coef, 1) else '' for coef in coefs]

    equation = f'Y = {coef_str[0]}'
    terms = ['Intercept', f"{factor_names['FactorA']}_num", f"{factor_names['FactorB']}_num",
             f"{factor_names['FactorA']}_num:{factor_names['FactorB']}_num"]

    for coef, term in zip(coef_str[1:], terms[1:]):
        equation += f' + {coef}*{term}'

    return equation   
    
def factorial_twolevels():
    st.title("Introduction to Factorial Designs")
    st.markdown("By **Leonardo H. Talero-Sarmiento** [View profile](https://apolo.unab.edu.co/en/persons/leonardo-talero)")
    st.markdown("A factorial design is an experimental design that studies the effects of two or more factors, each with multiple levels, on a dependent variable. In a factorial design, the factors are manipulated independently of each other, and the levels of each factor are crossed with the levels of the other factors. This allows researchers to study the main effects of each factor, as well as the interactions between the factors.")
    st.markdown("Factorial designs are more efficient than one-factor-at-a-time designs, because they allow researchers to study the effects of multiple factors with the same number of participants. Additionally, factorial designs can help researchers to identify interactions between factors, which can be important for understanding the effects of the factors on the dependent variable.")
    st.markdown("There are two main types of factorial designs: full factorial designs and fractional factorial designs. Full factorial designs include all possible combinations of levels for each factor. Fractional factorial designs are a subset of full factorial designs that include only a subset of the possible combinations of levels. Fractional factorial designs are less efficient than full factorial designs, but they can be used when resources are limited.")
    st.subheader('Two factos and two levels')
    
    # Create dictionary for custom factor names
    two_level_factor_names = {
        "FactorA": "FactorA",
        "FactorB": "FactorB"
    }
    
        # Assign new names
    
    two_level_factor_names["FactorA"] = st.sidebar.text_input("Enter a custom name for Factor A", "FactorA")
    two_level_factor_names["FactorB"] = st.sidebar.text_input("Enter a custom name for Factor B", "FactorB")
       
    coefficients = [
        st.sidebar.slider('Coefficient 0 (Intercept)', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 1 ({two_level_factor_names["FactorA"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 2 ({two_level_factor_names["FactorB"]})', -10.0, 10.0, 0.0),
        st.sidebar.slider(f'Coefficient 3 ({two_level_factor_names["FactorA"]}*{two_level_factor_names["FactorB"]})', -10.0, 10.0, 0.0)
    ]
    
    df = create_two_level_dataframe()

    # rename the columns in the DataFrame to match the user's custom names
    df.rename(columns={f"{factor}_num": f"{two_level_factor_names[factor]}_num" for factor in ["FactorA", "FactorB"]}, inplace=True)
    
    
    # Compute Y
    df = compute_twolevel_y(df, coefficients, two_level_factor_names)
    
    st.subheader('Dataframe')
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button(
        "Download data as CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv")

    # Plot 3D
    st.subheader('Factors space')
    # Plot
    plot_2d(df, two_level_factor_names)
    

    # Box plot
    st.subheader('Analysis of Y based on a source of variability')
    st.subheader('Box plot')
    st.markdown("*The Usefulness of Boxplots by Group in Data Analysis*")
    st.markdown("Creating a boxplot by group is a valuable technique in data analysis because it provides a visual summary of the distribution of a continuous variable across different categories or groups. It offers several benefits and can yield insightful observations. Here are a few reasons why making a boxplot by group is useful and the expected insights it can provide:")

    st.markdown("1. Comparison of Distributions")
    st.markdown("By grouping the data and creating boxplots for each group, you can easily compare the distributions of the variable across different categories. This comparison allows you to identify similarities, differences, or variations in the data between groups. For example, you can determine if there are differences in income distributions between different occupations or if there are variations in customer satisfaction scores across different regions.")

    st.markdown("2. Identifying Outliers")
    st.markdown("Boxplots display outliers as individual data points beyond the whiskers, which represent the range of typical values. Outliers can indicate anomalies or extreme values that deviate significantly from the rest of the data. By creating boxplots by group, you can quickly identify if certain categories have more outliers or if specific groups exhibit extreme values. This information can help uncover interesting patterns or potential data quality issues.")

    st.markdown("3. Assessing Skewness and Symmetry")
    st.markdown("The boxplot's visual representation allows you to assess the symmetry and skewness of the distribution within each group. By examining the relative positions of the median, lower quartile (25th percentile), and upper quartile (75th percentile), you can gain insights into the shape of the distribution. For instance, if the median is closer to the lower or upper quartile, it indicates skewness in that group's data, suggesting a non-normal distribution.")

    st.markdown("4. Detecting Shifts or Changes")
    st.markdown("Comparing boxplots across different groups or categories can help you identify shifts or changes in the distribution of a variable. If the medians or quartiles vary noticeably between groups, it suggests differences in the central tendency or spread of the data. This insight can be particularly valuable when analyzing trends over time, geographic regions, or different experimental conditions.")

    st.markdown("5. Exploring Relationships between Variables")
    st.markdown("Boxplots can be employed to explore relationships between a continuous variable and one or more categorical variables. By creating boxplots for different categories or groups, you can examine how the distribution of the continuous variable varies across the categorical variables. This analysis can provide initial insights into potential relationships and guide further investigations, such as conducting statistical tests or developing predictive models.")

    st.markdown("Overall, creating boxplots by group enhances the interpretability of data by providing a visual representation of the distributional characteristics within each category or group. It facilitates comparisons, helps identify outliers and patterns, and assists in understanding relationships between variables. These insights are essential for making informed decisions, identifying trends, and gaining a deeper understanding of the underlying data.")
    groupby_options = list(two_level_factor_names.values()) + [f'{i}*{j}' for i, j in combinations(two_level_factor_names.values(), 2)]
    groupby = st.selectbox('Group by', groupby_options, index=0)

    if groupby == f'{two_level_factor_names["FactorA"]}*{two_level_factor_names["FactorB"]}':
        df['Interaction'] = df[f'{two_level_factor_names["FactorA"]}_num'].astype(str) + '*' + df[f'{two_level_factor_names["FactorB"]}_num'].astype(str)
    else:
        df['Interaction'] = df[f'{groupby}_num']

    plot_boxplot(df, 'Interaction')
    
    # Surface plot
    st.subheader('Surface plot')
    st.markdown("*The Usefulness of Surface Plots in Two-Factorial Analysis*")
    st.markdown("Creating surface plots during a two-factorial analysis is a valuable technique in data analysis because it provides a visual representation of the interaction between two independent variables and their effect on a dependent variable. It offers several benefits and can yield insightful observations. Here are a few reasons why making a surface plot in a two-factorial analysis is useful and the expected insights it can provide:")

    st.markdown("1. Visualization of Interaction Effects")
    st.markdown("A two-factorial analysis aims to understand how two independent variables interact and influence a dependent variable. By creating a surface plot, you can visualize the interaction effects between the two factors. The surface plot displays the values of the dependent variable on a three-dimensional surface, where the x and y axes represent the levels of the two independent variables, and the z-axis represents the value of the dependent variable. This visualization allows you to observe how the levels of one factor may affect the relationship between the other factor and the dependent variable.")

    st.markdown("2. Identification of Optimal Conditions")
    st.markdown("Surface plots can help identify optimal conditions or combinations of the two independent variables that result in the highest or lowest values of the dependent variable. By analyzing the shape of the surface plot, you can determine the regions where the dependent variable is maximized or minimized. This information is valuable for making informed decisions or recommendations, such as identifying the best parameter settings in a manufacturing process or determining the ideal combination of factors in an experimental design.")

    st.markdown("3. Detection of Nonlinear Relationships")
    st.markdown("Surface plots are particularly useful for detecting nonlinear relationships between the independent variables and the dependent variable. If the surface plot exhibits curvatures or patterns that deviate from a simple linear relationship, it suggests the presence of nonlinear interactions. This insight can guide further analysis and modeling, helping to capture the complex relationships between variables more accurately.")

    st.markdown("4. Exploration of Response Surfaces")
    st.markdown("Surface plots allow for the exploration of response surfaces, which provide a comprehensive view of the relationship between the independent variables and the dependent variable. By examining the contour lines or gradients on the surface plot, you can gain insights into the magnitude and direction of changes in the dependent variable as the levels of the independent variables vary. This exploration facilitates a deeper understanding of the factors influencing the outcome and can guide further investigation or optimization efforts.")

    st.markdown("5. Communication of Findings")
    st.markdown("Surface plots are visually compelling and provide an effective means of communicating the results of a two-factorial analysis. They allow researchers, decision-makers, or stakeholders to grasp the complex relationships between variables in a more intuitive manner. The three-dimensional representation of the data aids in conveying the patterns, trends, and interactions in a visually appealing and easily interpretable format.")

    st.markdown("In summary, creating surface plots during a two-factorial analysis enhances the understanding of the interaction effects between independent variables and their impact on a dependent variable. They enable the visualization of complex relationships, identification of optimal conditions, detection of nonlinearities, exploration of response surfaces, and effective communication of findings. These insights are crucial for making informed decisions, optimizing processes, and gaining a deeper understanding of the underlying data.")
    plot_surface_twolevels(df, two_level_factor_names)

    # Model fitting
    st.subheader('Model fitting')


    # Similarly, when you call the fit_model and print_equation functions
    results = fit_model_twolevels(df, two_level_factor_names)
    st.latex(print_equation_twolevels(results, two_level_factor_names))

    st.text(results.summary())


    st.markdown("*Tips for Analyzing the `model.summary()` of Ordinary Least Squares (OLS) in Statsmodels*")
    st.markdown("Analyzing the `model.summary()` output from the OLS regression in Statsmodels is an important step in understanding the statistical significance and goodness of fit of the model. It provides comprehensive information about the model coefficients, standard errors, p-values, and various statistical measures. Here are some tips to effectively interpret the `model.summary()` output: ")

    st.markdown("1. Understand the Coefficients")
    st.markdown("The coefficients represent the estimated effects of the independent variables on the dependent variable. Pay attention to the sign (positive or negative) of the coefficients, as it indicates the direction of the relationship. A positive coefficient implies a positive relationship, while a negative coefficient suggests a negative relationship.")

    st.markdown("2. Assess the Statistical Significance")
    st.markdown("The p-values associated with each coefficient indicate the statistical significance of the corresponding variable. Lower p-values (typically below 0.05) suggest stronger evidence against the null hypothesis, indicating a significant relationship between the variable and the response. Focus on variables with low p-values when interpreting the model's importance.")

    st.markdown("3. Evaluate the Confidence Intervals")
    st.markdown("The confidence intervals provide a range of plausible values for the coefficients. The wider the interval, the greater the uncertainty in the estimate. Narrow confidence intervals indicate more precise estimates. Look for intervals that do not include zero, as this suggests a statistically significant relationship.")

    st.markdown("4. Examine the R-squared")
    st.markdown("The R-squared value measures the proportion of the variance in the dependent variable that can be explained by the independent variables. A higher R-squared indicates a better fit of the model to the data. However, it's important to consider the context of the analysis and the specific field of study, as what constitutes a high or acceptable R-squared can vary.")

    st.markdown("5. Check for Residual Analysis")
    st.markdown("The model summary provides insights into the residuals, which are the differences between the observed and predicted values. Assess the residuals for any patterns or systematic deviations, such as nonlinearity, heteroscedasticity, or autocorrelation. Plotting the residuals against the predicted values or independent variables can help identify potential issues.")

    st.markdown("6. Consider Model Assumptions")
    st.markdown("OLS regression relies on certain assumptions, such as linearity, independence, homoscedasticity, and normality of residuals. Examine diagnostic tests, such as the Jarque-Bera test for normality or tests for heteroscedasticity, to ensure that these assumptions hold. Violations of assumptions may affect the validity of the model.")

    st.markdown("7. Compare with Domain Knowledge")
    st.markdown("While statistical measures are important, it is equally crucial to interpret the results in the context of the specific domain or field of study. Consider the practical significance of the coefficients and the overall fit of the model in relation to the underlying subject matter. Expert knowledge can provide valuable insights that complement the statistical analysis.")

    st.markdown("By following these tips, you can effectively analyze the `model.summary()` output of the OLS regression in Statsmodels and gain a deeper understanding of the model's performance, significance, and potential limitations.")

    

pages = {
    "Introduction to Factorial Designs": factorial_twolevels,
    "Factorial designs with trhee factos and tree levels": Three_factorial
    
}

st.title('Navigation')
selection = st.radio("Go to", list(pages.keys()))

# call the function to draw the selected page
pages[selection]()

