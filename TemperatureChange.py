import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("World Temperature")

    # Create a sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Framework and Pre-processing", "Visualization", "Machine Learning Methods", "Prediction", "Conclusion"))

    if page == "Home":
        home_page()
    elif page == "Framework and Pre-processing":
        framework_and_preprocessing_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Machine Learning Methods":
        machine_learning_methods_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Conclusion":
        conclusion_page()

def home_page():
    st.write("Welcome to the World Temperature app!")

    st.header("Context")
    st.write("The project is based on a general global problem regarding the currently experienced and predicted global warming and its dependencies and influences by greenhouse gas emissions and especially CO2 emissions.")
    st.write("Underlying for this project are two different sets of data. The first one is providing information on the global GHG and therefore also CO2 emissions over a certain period including the resulted change in temperature caused by these emissions. The second dataset provides just information on the temperature change in the past without any further connection or breakdown into the underlying emissions.")
    st.write("Even though this project has no direct connection to a certain business case of one of the participants, it is somehow connected to every business case due to its omnipresence in politics and society, which results in side effects for businesses.")
    st.write("From a scientific perspective, the project will not focus on any special kind of climate change theories or models but will only focus on the technical aspects of data analysis and machine learning with the aim to predict future development based on different models trained with existing data from the past.")

    # Embedding the picture
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Global_Warming_Predictions_Map.jpg", caption="Global Warming Predictions Map", use_column_width=True)

def framework_and_preprocessing_page():
    st.write("Welcome to the Framework and Pre-processing page!")

    st.header("Framework")

    st.write("The following datasets have been used during the project:")

    st.subheader("[Temperature Change data - FAOSTAT_data_1-10-2022.csv](https://www.kaggle.com/datasets/sevgisarac/temperature-change/?select=FAOSTAT_data_1-10-2022.csv)")
    st.write("This dataset is freely available on Kaggle.com. The FAOSTAT Temperature Change domain disseminates statistics of mean surface temperature change by country, with annual updates. The current dissemination covers the period 1961–2020.")
    st.write("The dataset has 229,925 entries and 14 columns. It is totally clear.")
    image_path = "/workspaces/temperature-change---dataanalyst/pictures/temp_head.PNG"
    st.image(image_path, caption='First 10 lines of the temperature change dataset.', use_column_width=True)
  
 
    st.subheader("[Emission and Temperature data - owid-co2-data.csv](https://github.com/owid/co2-data)")
    st.write("This dataset is freely available on GitHub. This CO2 and Greenhouse Gas Emissions dataset is a collection of key metrics maintained by 'Our World in Data'. It is uploaded regularly and includes data on CO2 emission (annual, per capita, cumulative and consumption-based), other greenhouse gases, energy mix and other relevant metrics.")
    image_path = "/workspaces/temperature-change---dataanalyst/pictures/co2_head.PNG"
    st.image(image_path, caption='First 10 lines of the co2-emissions dataset. Only the first frew columns of in total 79 columns are shown.', use_column_width=True)

def visualization_page():
    
    st.header("Data Visualization")
 
    st.write("This page provides on overview of different visualizations of the used dataset, that were generated during the data cleaning and pre-processing. These charts help to understand the datasets, their structure and limitations. They also show extreme values as well as outliers that would be unrealistic.")
    
    # Displaying different visualizations
    image_path = "/workspaces/temperature-change---dataanalyst/pictures/Screenshot2.PNG"
    st.image(image_path, caption='Development of the CO2-emissions within the dataset.', use_column_width=True)
    st.write("The chart above pictures the CO2-emission data throughout the years. Data is available from 1990 until 2020. It shows the development throughout the years and also shows potential outliers or extreme values (which isn't the case here).")
    
    image_path_2 = "/workspaces/temperature-change---dataanalyst/pictures/Screenshot3.PNG"
    st.image(image_path_2, caption='Distribution of the CO2-emissions within the dataset.', use_column_width=True)
    st.write("The second chart shows the development of the temperature change throughout the years. Here data is available from 1960 until 2020. For this longer timeframe the scatter plot better shows outliers and extreme values as it picutres every single data point without any connection.")
    
    image_path_3 = "/workspaces/temperature-change---dataanalyst/pictures/Correlation.png"
    st.image(image_path_3, caption='Correlation between the different variables in the CO2-emissions dataset.', use_column_width=True)
    st.write("This chart show the correlation of the different variables within the dataset. It can be seen that there is a strong correlation between the CO2 emissions and the temperature change.")
    	

def machine_learning_methods_page():

    st.title("Machine Learning Methods")    

    st.header("Linear Regression Model")
    st.write("Linear regression is a statistical method used to model the relationship between a dependent variable (CO2 emissions or temperature change) and one or more independent variables (year). It is one of the simplest and most commonly used regression techniques. Linear regression assumes that there is a linear relationship between the independent variables and the dependent variable. The model finds the best-fitting straight line to describe the relationship between the variables. This line is determined by minimizing the sum of the squared differences between the observed values and the values predicted by the model. In our specific use case, linear regression is employed to predict future CO2 emissions based on historical data. By fitting a linear regression model to the historical CO2 emissions data, we can estimate the relationship between CO2 emissions and time (year). This allows us to make predictions about future CO2 emissions based on the observed trend in the data. The linear regression model calculates the coefficients of the linear equation that best fits the data, enabling us to forecast CO2 emissions for future years.")

    # Load the dataset from the GitHub raw URL
    url = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'
    df_co2 = pd.read_csv(url)

    # Clean the dataframe
    df_co2 = df_co2.drop(['country', 'population', 'gdp', 'cement_co2', 'cement_co2_per_capita', 'co2_growth_abs', 'co2_growth_prct',
                          'co2_including_luc_growth_abs', 'co2_including_luc', 'co2_including_luc_growth_prct', 'co2_including_luc_per_capita', 'co2_including_luc_per_gdp',
                          'co2_including_luc_per_unit_energy', 'co2_per_capita', 'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2', 'coal_co2_per_capita',
                          'consumption_co2', 'consumption_co2_per_capita', 'consumption_co2_per_gdp', 'cumulative_cement_co2', 'cumulative_co2',
                          'cumulative_co2_including_luc', 'cumulative_coal_co2', 'cumulative_flaring_co2', 'cumulative_gas_co2', 'cumulative_luc_co2',
                          'cumulative_oil_co2', 'cumulative_other_co2', 'energy_per_capita', 'energy_per_gdp', 'flaring_co2', 'flaring_co2_per_capita', 'gas_co2', 'gas_co2_per_capita',
                          'ghg_excluding_lucf_per_capita', 'ghg_per_capita', 'land_use_change_co2', 'land_use_change_co2_per_capita', 'methane', 'methane_per_capita', 'nitrous_oxide',
                          'nitrous_oxide_per_capita', 'oil_co2', 'oil_co2_per_capita', 'other_co2_per_capita', 'other_industry_co2', 'primary_energy_consumption',
                          'share_global_cement_co2', 'share_global_co2', 'share_global_co2_including_luc', 'share_global_coal_co2', 'share_global_cumulative_cement_co2',
                          'share_global_cumulative_co2', 'share_global_cumulative_co2_including_luc', 'share_global_cumulative_coal_co2',
                          'share_global_cumulative_flaring_co2', 'share_global_cumulative_luc_co2', 'share_global_cumulative_gas_co2', 'share_global_luc_co2',
                          'share_global_oil_co2', 'share_global_cumulative_oil_co2', 'share_global_cumulative_other_co2', 'share_global_flaring_co2', 'share_global_gas_co2',
                          'share_global_other_co2', 'share_of_temperature_change_from_ghg', 'temperature_change_from_ch4', 'temperature_change_from_ghg',
                          'temperature_change_from_n2o', 'total_ghg', 'total_ghg_excluding_lucf', 'trade_co2', 'trade_co2_share'], axis=1)

    # Filter the dataframe
    df_co2 = df_co2.loc[df_co2['year'] > 1989]  # Only keep years from 1990 onwards to fit together with the temperature change dataset
    df_co2 = df_co2.loc[df_co2['year'] < 2021]  # Only keep years up until 2020 (included) to fit together with the temperature change dataset

    # Filter for Canada (CAN)
    df_co2_can = df_co2[df_co2['iso_code'] == 'CAN']

    # Display the filtered dataframe
    st.subheader("Filtered Dataset for Canada (CAN)")
    st.write(df_co2_can)

    # Perform 20% train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_co2_can[['year']], df_co2_can['co2'], test_size=0.2, random_state=0)

    # Train the linear regression model for CO2 emissions prediction
    model_co2 = LinearRegression()
    model_co2.fit(X_train, y_train)

    # Predict CO2 emissions for the same timeframe
    y_pred_co2 = model_co2.predict(df_co2_can[['year']])

    st.write("1. **Actual vs Predicted CO2 Emissions for Canada (CAN)**: The chart illustrates the comparison between actual CO2 emissions in Canada, represented by blue dots, and the predicted emissions based on the linear regression model, depicted by the red line. Overall, the model accurately captures the general trend of CO2 emissions, although there are occasional deviations between the actual and predicted values. While the model serves as a valuable tool for understanding and forecasting future CO2 emission trends in Canada, these discrepancies suggest potential factors not fully accounted for in the model, emphasizing the importance of ongoing refinement and validation.")
    # Calculate metrics for CO2 emissions prediction
    mse_co2 = mean_squared_error(df_co2_can['co2'], y_pred_co2)
    r2_co2 = r2_score(df_co2_can['co2'], y_pred_co2)

    st.write(f"**Metrics for CO2 Emissions Prediction:**")
    st.write(f"- Mean Squared Error (MSE): {mse_co2}")
    st.write(f"- R-squared (R2) Score: {r2_co2}")

    # Plot actual and predicted CO2 emissions
    plt.figure(figsize=(10, 6))
    plt.scatter(df_co2_can['year'], df_co2_can['co2'], color='blue', label='Actual CO2 Emissions')
    plt.plot(df_co2_can['year'], y_pred_co2, color='red', label='Predicted CO2 Emissions')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions')
    plt.title('Actual vs Predicted CO2 Emissions for Canada (CAN)')
    plt.legend()
    st.pyplot(plt)

    # Train the linear regression model for temperature change prediction based on CO2 emissions
    model_temp = LinearRegression()
    model_temp.fit(df_co2_can[['co2']], df_co2_can['temperature_change_from_co2'])

    # Predict temperature change based on CO2 emissions
    y_pred_temp = model_temp.predict(df_co2_can[['co2']])

    st.write("2. **Actual vs Predicted Temperature Change based on CO2 Emissions for Canada (CAN)**: This visualization contrasts the actual temperature changes associated with CO2 emissions in Canada (blue dots) with the predicted temperature changes derived from the linear regression model (red dots). Both actual and predicted temperature changes exhibit a positive correlation with CO2 emissions, indicating the significant impact of CO2 on temperature. While the model successfully captures the overall relationship between CO2 emissions and temperature change, variations in actual temperature changes highlight the complexity of climate dynamics and the need for comprehensive climate mitigation strategies.")

    # Calculate metrics for temperature change prediction
    mse_temp = mean_squared_error(df_co2_can['temperature_change_from_co2'], y_pred_temp)
    r2_temp = r2_score(df_co2_can['temperature_change_from_co2'], y_pred_temp)

    st.write(f"**Metrics for Temperature Change Prediction:**")
    st.write(f"- Mean Squared Error (MSE): {mse_temp}")
    st.write(f"- R-squared (R2) Score: {r2_temp}")

    # Plot actual and predicted temperature change based on CO2 emissions
    plt.figure(figsize=(10, 6))
    plt.scatter(df_co2_can['co2'], df_co2_can['temperature_change_from_co2'], color='blue', label='Actual Temperature Change')
    plt.plot(df_co2_can['co2'], y_pred_temp, color='red', label='Predicted Temperature Change')
    plt.xlabel('CO2 Emissions')
    plt.ylabel('Temperature Change')
    plt.title('Actual vs Predicted Temperature Change based on CO2 Emissions for Canada (CAN)')
    plt.legend()
    st.pyplot(plt)

    st.write("3. **Actual vs Predicted Temperature Change for Canada (CAN)**: In this chart, the actual temperature changes recorded in Canada (blue dots) are juxtaposed with the predicted temperature changes based on the linear regression model (red line). The model offers a straightforward prediction of future temperature changes solely based on historical data, showing a linear trend over time. While the model captures the overall increasing trend in temperature, it may oversimplify the multifaceted factors influencing climate change. Incorporating additional variables and advanced modeling techniques could enhance the accuracy of temperature predictions and provide more robust insights for climate policy formulation and implementation.")

    # Plot actual and predicted temperature change
    plt.figure(figsize=(10, 6))
    plt.scatter(df_co2_can['year'], df_co2_can['temperature_change_from_co2'], color='blue', label='Actual Temperature Change')
    plt.scatter(df_co2_can['year'], y_pred_temp, color='red', label='Predicted Temperature Change')
    plt.xlabel('Year')
    plt.ylabel('Temperature Change')
    plt.title('Actual vs Predicted Temperature Change for Canada (CAN)')
    plt.legend()
    st.pyplot(plt)

    st.header("Prophet")

    st.write("""
    Prophet, developed by Facebook, is a robust statistical tool designed for time series forecasting. It's particularly adept at modeling the relationship between a dependent variable (such as temperature change) and time, while accommodating various patterns like seasonality and holiday effects. Unlike linear regression, Prophet does not rely on the assumption of a linear relationship between the variables.
    Prophet works by decomposing time series data into several components, including trend, seasonality, and holidays. It then utilizes a flexible Bayesian framework to model these components and generate forecasts. The model's parameters are estimated using a procedure that simultaneously considers both the trend and the seasonal patterns present in the data.
    In our specific scenario, Prophet can be employed to forecast future temperature changes based on historical data. By fitting the Prophet model to the historical temperature data, we can capture the underlying trends and seasonal patterns in temperature variations over time. This enables us to make accurate predictions about future temperature changes, taking into account factors such as yearly trends and any seasonal fluctuations.
    Prophet's ability to handle non-linear relationships and complex seasonal patterns makes it well-suited for forecasting temperature changes, which often exhibit such characteristics. By leveraging the capabilities of Prophet, we can obtain reliable forecasts of temperature changes for future periods, aiding in various applications such as climate research, urban planning, and agriculture.
    """)

    image_path = "/workspaces/temperature-change---dataanalyst/pictures/Prophet.PNG"
    st.image(image_path, caption='Prediction of temperature change using Prophet.', use_column_width=True)


def prediction_page():
    st.title("Prediction")

    st.write("Welcome to the Prediction page!")

    st.header("Linear Regression Model for Predictions")
    st.write("Linear regression is used to predict future CO2 emissions and temperature change based on historical data.")

    # Load the dataset from the GitHub raw URL
    url = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'
    df_co2 = pd.read_csv(url)

    # Clean the dataframe
    df_co2 = df_co2.drop(['country', 'population', 'gdp', 'cement_co2', 'cement_co2_per_capita', 'co2_growth_abs', 'co2_growth_prct',
                          'co2_including_luc_growth_abs', 'co2_including_luc', 'co2_including_luc_growth_prct', 'co2_including_luc_per_capita', 'co2_including_luc_per_gdp',
                          'co2_including_luc_per_unit_energy', 'co2_per_capita', 'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2', 'coal_co2_per_capita',
                          'consumption_co2', 'consumption_co2_per_capita', 'consumption_co2_per_gdp', 'cumulative_cement_co2', 'cumulative_co2',
                          'cumulative_co2_including_luc', 'cumulative_coal_co2', 'cumulative_flaring_co2', 'cumulative_gas_co2', 'cumulative_luc_co2',
                          'cumulative_oil_co2', 'cumulative_other_co2', 'energy_per_capita', 'energy_per_gdp', 'flaring_co2', 'flaring_co2_per_capita', 'gas_co2', 'gas_co2_per_capita',
                          'ghg_excluding_lucf_per_capita', 'ghg_per_capita', 'land_use_change_co2', 'land_use_change_co2_per_capita', 'methane', 'methane_per_capita', 'nitrous_oxide',
                          'nitrous_oxide_per_capita', 'oil_co2', 'oil_co2_per_capita', 'other_co2_per_capita', 'other_industry_co2', 'primary_energy_consumption',
                          'share_global_cement_co2', 'share_global_co2', 'share_global_co2_including_luc', 'share_global_coal_co2', 'share_global_cumulative_cement_co2',
                          'share_global_cumulative_co2', 'share_global_cumulative_co2_including_luc', 'share_global_cumulative_coal_co2',
                          'share_global_cumulative_flaring_co2', 'share_global_cumulative_luc_co2', 'share_global_cumulative_gas_co2', 'share_global_luc_co2',
                          'share_global_oil_co2', 'share_global_cumulative_oil_co2', 'share_global_cumulative_other_co2', 'share_global_flaring_co2', 'share_global_gas_co2',
                          'share_global_other_co2', 'share_of_temperature_change_from_ghg', 'temperature_change_from_ch4', 'temperature_change_from_ghg',
                          'temperature_change_from_n2o', 'total_ghg', 'total_ghg_excluding_lucf', 'trade_co2', 'trade_co2_share'], axis=1)

    # Filter the dataframe
    df_co2 = df_co2.loc[df_co2['year'] > 1989]  # Only keep years from 1990 onwards to fit together with the temperature change dataset
    df_co2 = df_co2.loc[df_co2['year'] < 2021]  # Only keep years up until 2020 (included) to fit together with the temperature change dataset

    # User input for predicted timeframe
    st.header("Predicted Timeframe")
    start_year = st.number_input("Enter the start year:", min_value=2025, max_value=2100, value=2025)
    end_year = st.number_input("Enter the end year:", min_value=2026, max_value=2100, value=2030)

    # User input for countries
    st.header("Select Countries")
    selected_countries = st.multiselect("Select G7 countries:", ['CAN', 'FRA', 'DEU', 'ITA', 'JPN', 'GBR', 'USA'])

    # Definition of Future prediction timeline
    Future = np.arange(start_year, end_year + 1).reshape(-1, 1)

    # Initialize a color cycle for plotting
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

    # Plot CO2 emission for selected countries
    plt.figure(figsize=(10, 6))

    for iso in selected_countries:
        df_country = df_co2[df_co2['iso_code'] == iso]

        X_co2 = df_country.drop({'iso_code', 'co2', 'temperature_change_from_co2'}, axis=1)
        Y_co2 = df_country['co2']

        # 20% Train/Test-split for CO2-prediction based on year
        X_co2_train, X_co2_test, Y_co2_train, Y_co2_test = train_test_split(X_co2, Y_co2, test_size=0.2, shuffle=False)
        # Linear regression model
        lin_reg_co2 = LinearRegression()
        lin_reg_co2.fit(X_co2_train.values, Y_co2_train.values)

        # CO2 predictions
        co2_predictions = lin_reg_co2.predict(Future)
        co2_predictions = co2_predictions.reshape(-1, 1)

        # Plot CO2 emission predictions
        plt.plot(Future, co2_predictions, label=f"{iso} - CO2 Emissions")

    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions")
    plt.title("Future Prediction of CO2 Emissions")
    plt.legend()
    st.pyplot(plt)

    # Plot temperature change for selected countries
    plt.figure(figsize=(10, 6))

    for iso in selected_countries:
        df_country = df_co2[df_co2['iso_code'] == iso]

        X_temp = df_country.drop({'year', 'iso_code', 'temperature_change_from_co2'}, axis=1)
        Y_temp = df_country['temperature_change_from_co2']

        # 20% Train/Test-split for Temperature prediction based on CO2-emission
        X_temp_train, X_temp_test, Y_temp_train, Y_temp_test = train_test_split(X_temp, Y_temp, test_size=0.2, shuffle=False)
        # Linear regression model
        lin_reg_temp = LinearRegression()
        lin_reg_temp.fit(X_temp_train.values, Y_temp_train.values)

        # CO2 predictions
        co2_predictions = lin_reg_co2.predict(Future)
        co2_predictions = co2_predictions.reshape(-1, 1)
        # Temperature predictions
        temp_predictions = lin_reg_temp.predict(co2_predictions)

        # Plot temperature change predictions
        plt.plot(Future, temp_predictions, label=f"{iso} - Temperature Change")

    plt.xlabel("Year")
    plt.ylabel("Temperature Change")
    plt.title("Future Prediction of Temperature Change")
    plt.legend()
    st.pyplot(plt)

def conclusion_page():
    st.title("Conclusion")

    # Exploring the datasets and preprocessing
    st.write("This project aimed to analyze the relationship between CO2 emissions and temperature change using machine learning techniques. We started by exploring the datasets, which provided information on global greenhouse gas emissions and temperature changes over time. During the preprocessing, the data has been analyzed, with a special focus on relevant data, missing values, and correlations.")

    # Testing machine learning algorithms
    st.write("After that, the linear regression model as well as Facebook’s Prophet has been used as machine learning algorithms to test the prediction of CO emissions and temperature changes for one single country.")

    # Setting up flexible future prediction for G7 countries
    st.write("Based on these tests, a flexible future prediction for the G7 countries has been set up. Based on the input of which countries and future timeframe should be predicted, the linear regression models were trained to predict CO2 emissions and use this prediction to predict the expected temperature change.")

    # Limitations of linear regression in predicting CO2 emissions and temperature changes
    st.write("Linear regression provides a simple prediction based on a linear trend. For complex topics like CO2 emissions and temperature change, it quickly reaches its limitations. These two outcomes depend on many different factors. This leads to confusing outcomes like sinking emissions with still rising temperature. This might be due to other influences on temperature change apart from CO emissions, or the possibility that reduced emissions might take a longer time until it results in reduced temperature change. Both factors are not known and can also not be implemented with a linear model.")

    # Conclusion about the project
    st.write("Overall, the project and its Streamlit application show the successful application of basic data processing, analysis, and prediction, which was the main aim of the project.")

if __name__ == "__main__":
    main()
