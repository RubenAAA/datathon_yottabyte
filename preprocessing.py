# Importing all the necessary packages
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import pandas as pd
from pandas.errors import PerformanceWarning
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima.arima import ndiffs, nsdiffs
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor


# For legibility, we mute some warnings
import warnings

# Ignore FutureWarning for deprecated 'T' frequency in Prophet
warnings.filterwarnings("ignore", category=FutureWarning, message="'T' is deprecated")

# Ignore PerformanceWarning from pandas
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Mix of different sources, mostly ESO
balancing_df = pd.read_csv("balancing_data.csv")
# Demand data only for GB
GB_demand_df = pd.read_csv("demand_load_data.csv")
# Generation data only for GB
GB_generation_df = pd.read_csv("generation_data.csv")
# the price dataframe only concerns EPEX (only prices from there)
EPEX_price_df = pd.read_csv("price_data.csv")
def rename_balancing_columns(df):
    # Define a dictionary for concise renaming
    rename_map = {
        'GMT Time': 'GMT Time',
        'System Price (ESO Outturn) - GB (£/MWh)': 'System_Price',
        'NIV Outturn (+ve long) - GB (MW)': 'NIV_Outturn',
        'BM Bid Acceptances (total) - GB (MW)': 'BM_Bid_Acceptances',
        'BM Offer Acceptances (total) - GB (MW)': 'BM_Offer_Acceptances',
        'Total BSAD Volume - Turn Up - GB (MW)': 'BSAD_Turn_Up',
        'Total BSAD Volume - Turn Down - GB (MW)': 'BSAD_Turn_Down',
        'Total BSAD Volume - Total - GB (MW)': 'BSAD_Total',
        'Intraday Volume (EPEX Outturn, APX, MID) - GB (MWh)': 'EPEX_Intraday_Volume'
    }
    
    # Apply the renaming map
    df = df.rename(columns=rename_map)

    # Force all the non datetime columns to numeric
    for column in df.columns:
        if column != 'GMT Time':  # Skip the 'GMT Time' column
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Apply the function to rename columns in balancing_df
balancing_df = rename_balancing_columns(balancing_df)

print("Final columns:")
print(balancing_df.columns.values)

def rename_demand_columns(df):
    """
    Rename columns for easier reference and convert non-datetime columns to numeric.
    """
    # Define a dictionary for concise renaming
    rename_map = {
        'GMT Time': 'GMT Time',
        'Loss of Load Probability - Latest - GB ()': 'Loss_of_Load_Prob',
        'Actual Total Load - GB (MW)': 'Total_Load',
        'Demand Outturn (ITSDO) - GB (MW)': 'Demand_Outturn'
    }
    
    # Apply the renaming map
    df = df.rename(columns=rename_map)

    # Force all the non-datetime columns to numeric
    for column in df.columns:
        if column != 'GMT Time':  # Skip the 'GMT Time' column
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Apply the renaming and filling functions
GB_demand_df = rename_demand_columns(GB_demand_df)


print("Final columns:")
print(GB_demand_df.columns.values)
def rename_columns_generation(df):
    # Define a function to clean each column name
    def clean_column_name(col):
        # Extract the generation type using regex
        match = re.search(r'Actual Aggregated Generation By Type - (.+?) - GB', col)
        if match:
            # Replace spaces with underscores for readability
            return match.group(1).replace(" ", "_")
        return col  # Return the column as is if no match is found

    # Rename columns using the clean_column_name function
    df.columns = [clean_column_name(col) for col in df.columns]
    for column in df.columns:
        if column != 'GMT Time':  # Skip the 'GMT Time' column
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

# Apply the function to rename columns in generation_df
GB_generation_df = rename_columns_generation(GB_generation_df)


print("Final columns:")
print(GB_generation_df.columns.values)
def rename_epex_columns(df):
    # Define a dictionary for manual renaming based on your desired column names
    rename_map = {
        'GMT Time': 'GMT Time',
        'Day Ahead Price (EPEX half-hourly, local) - GB (LC/MWh)': 'Day_Ahead_Price',
        'Intraday Price (EPEX Outturn, APX, MID) - GB (£/MWh)': 'Intraday_Price'
    }

    # Rename columns using the dictionary
    df = df.rename(columns=rename_map)
    for column in df.columns:
        if column != 'GMT Time':  # Skip the 'GMT Time' column
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df

# Apply the function to rename columns in EPEX_price_df
EPEX_price_df = rename_epex_columns(EPEX_price_df)


print("Final columns:")
print(EPEX_price_df.columns.values)
# Set 'GMT Time' as index for each dataframe
balancing_df.set_index('GMT Time', inplace=True)
GB_demand_df.set_index('GMT Time', inplace=True)
GB_generation_df.set_index('GMT Time', inplace=True)
EPEX_price_df.set_index('GMT Time', inplace=True)

# Merge using index
merged_df = balancing_df.join([GB_demand_df, GB_generation_df, EPEX_price_df], how='inner')
# We put back the datetime column into the merged DF and rename it for practicality
merged_df.reset_index(inplace=True)
merged_df.rename(columns={'GMT Time': 'Datetime'}, inplace=True)

print("Merged columns:")
print(merged_df.columns.values)
def calculate_fft(df, variable, n_top_seasonalities, threshold_pc=0.02):
    """
    Calculate significant positive frequencies and their amplitudes using Fast Fourier Transform (FFT),
    selecting the lower of 2% of the max amplitude or the top `n` frequencies.

    Parameters:
    - df (DataFrame): The input DataFrame containing the time series data.
    - variable (str): The name of the column in `df` on which to perform FFT.
    - n_top_seasonalities (int): The maximum number of significant frequencies to consider.
    - threshold_pc (float): Percentage (0 < threshold_pc <= 1) of the maximum amplitude to filter significant frequencies.

    Returns:
    - zip: A generator yielding (positive frequency, amplitude) for each significant frequency.
    """
    # Compute fast Fourier transform
    price_fft = np.fft.fft(df[variable].dropna())

    # Get frequencies corresponding to FFT coefficients
    freqs = np.fft.fftfreq(len(price_fft), d=1/48)

    # Calculate amplitudes
    amplitudes = np.abs(price_fft)

    # Calculate the threshold based on 2% of the max amplitude
    threshold = threshold_pc * np.max(amplitudes)

    # Filter positive frequencies with amplitudes above threshold
    positive_indices = np.where((amplitudes > threshold) & (freqs > 0))
    positive_freqs = freqs[positive_indices]
    positive_amplitudes = amplitudes[positive_indices]

    # Sort by amplitude and select the lower of `n_top_seasonalities` or all significant frequencies
    sorted_indices = np.argsort(positive_amplitudes)[::-1]
    selected_indices = sorted_indices[:min(n_top_seasonalities, len(sorted_indices))]

    # Select the top frequencies and amplitudes
    significant_freqs = positive_freqs[selected_indices]
    print(len(significant_freqs))
    significant_amplitudes = positive_amplitudes[selected_indices]

    return zip(significant_freqs, significant_amplitudes)



def prophet_predictions(df, variable, freq_amp):
    """
    Generate predictions using Prophet with multiple seasonalities based on significant frequencies.

    This function applies Prophet to model and predict a specified variable, adding custom seasonalities 
    derived from significant frequencies (e.g., daily, weekly patterns). The seasonalities are added 
    dynamically based on the frequency components identified through FFT, with Fourier orders adjusted 
    for shorter and longer periods.

    Parameters:
    - df (DataFrame): The input DataFrame containing the time series data.
    - variable (str): The name of the column in `df` to be modeled by Prophet.
    - freq_amp (list of tuples): A list of (frequency, amplitude) pairs, where each frequency represents 
                                 a significant periodic component to be modeled as seasonality.

    Returns:
    - forecast (DataFrame): The forecasted values for the specified period, including trend and seasonal components.
    """
    # Use Prophet to model_F multiple seasonalities
    prophet_balancing_df = df.reset_index().rename(columns={'Datetime': 'ds', variable: 'y'})
    model_F = Prophet()

    # Adding seasonalities based on significant frequencies
    for freq, amp in freq_amp:
        if freq != 0:  # Ignore the DC component
            period_in_days = 1 / freq
            # Add seasonality to Prophet
            seasonality_name = f"seasonal_freq_{freq:.4f}"
            if period_in_days <= 1:
                fourier_order = 5
            elif period_in_days > 1 and period_in_days <= 7:
                fourier_order = 10
            else:
                fourier_order = 20
            model_F.add_seasonality(name=seasonality_name, period=period_in_days, fourier_order=fourier_order)

    # Fit the model_F
    model_F.fit(prophet_balancing_df)

    # Make future dataframe for predictions, 48 rows because we predict for next day
    future = model_F.make_future_dataframe(periods=48, freq='30T')

    forecast = model_F.predict(future)

    # Plot the forecast
    # model_F.plot(forecast)
    # plt.show()
    return forecast


def t_arima(df, p, q):
    """
    Fit an ARIMA model to the time series data and forecast future values.

    This function applies an ARIMA model to a specified variable within a DataFrame, determining the degree 
    of differencing (d) based on stationarity tests. It then forecasts future values beyond the length of 
    the data provided, accommodating additional time steps for further predictions.

    Parameters:
    - df (Series): The input time series data.
    - p (int): The order of the autoregressive part.
    - q (int): The order of the moving average part.

    Returns:
    - Series: The forecasted values over the extended range, including additional time steps.
    """
    # Fit ARIMA

    # setting the frequency for the arima
    # df = df.asfreq('30T')
    
    # TO CHECK:
    # Tests
    # s = 12
    d = ndiffs(df, alpha = 0.05, test='adf')  # regular differences?
    # D = nsdiffs(y, m = s, test='ocsb') # # seasonal differences?)

    arima_model = ARIMA(df.dropna(), order=(p, d, q))
    arima_fit = arima_model.fit()

    forecast = arima_fit.predict(start=0, end=len(df) - 1 + 48)  # we predict for the 48 rows after
    return pd.Series(forecast, index=df.index)


def a_arima(df):
    """
    Fit an ARIMA model to the time series data and forecast future values.

    This function uses auto_arima from the pmdarima package to automatically select the best ARIMA model
    based on AIC/BIC criteria. It performs both in-sample prediction and forecasts future values beyond
    the length of the data provided, tailored for data in half-hour increments.

    Parameters:
    - df (Series): The input time series data.

    Returns:
    - DataFrame: A DataFrame containing the in-sample predictions and forecasted values over an extended range.
    """
    # Ensure there are no NaNs and the series is a pandas Series
    df = pd.Series(df.dropna())

    # Fit ARIMA model automatically
    model = pm.auto_arima(df, seasonal=False, stepwise=True, suppress_warnings=True, 
                          error_action="ignore", trace=True, test="adf", d=None,
                          max_p=5, max_q=5)

    # In-sample predictions
    in_sample_predictions = model.predict_in_sample()

    # Forecasting 48 periods ahead (24 hours at 30-minute intervals)
    future_forecast, conf_int = model.predict(n_periods=48, return_conf_int=True)

    # Create a DataFrame to store in-sample predictions and forecasts
    index_of_fc = pd.date_range(df.index[-1], periods=49, freq='30T')[1:]  # Adjust freq='30T' for half-hourly
    combined = pd.DataFrame(index=df.index.append(index_of_fc), columns=['predictions'])

    # Insert in-sample predictions and forecasts into the DataFrame
    combined.loc[df.index, 'predictions'] = in_sample_predictions
    combined.loc[index_of_fc, 'predictions'] = future_forecast
    combined["predictions"] = combined["predictions"].astype('float64')
    return combined["predictions"]


def garch(residuals, alpha):
    test_stat, p_value, _, _ = het_arch(residuals.dropna())
    if p_value < 0.05:
        print("There are benefits of using an ARCH / GARCH model")
        # Assuming 'residuals' are from an ARIMA model
        residuals = residuals.dropna()

        best_aic = np.inf
        best_p = 1
        best_q = 1
        for p in range(1, 12):  # Autoregressive order
            for q in range(1, 12):  # Moving average order
                model = arch_model(residuals, vol='Garch', p=p, q=q)
                fitted_model = model.fit(disp='off')
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_p = p
                    best_q = q
        # Fit GARCH model on ARIMA residuals
        garch_model = arch_model(residuals, vol='Garch', p=best_p, q=best_q)
        garch_fit = garch_model.fit(disp="off")

        # Forecast GARCH variance (get predicted standard deviation for each time step)
        garch_forecast_var = garch_fit.forecast(horizon=48)
        garch_std_forward = np.sqrt(garch_forecast_var.variance.values[-1])  # This gives the volatility (std deviation) forecast

        # Get historical volatility (conditional standard deviation) on the fitted data
        garch_std_historical = garch_fit.conditional_volatility

        # Combine backcast and forecast for a 48-period range in each direction
        garch_std_combined = np.concatenate((garch_std_historical, garch_std_forward))

        
        #############################
        # Get residuals
        residuals = garch_fit.resid

        # Standardized residuals
        standardized_residuals = residuals / garch_fit.conditional_volatility
        
        # Define a scaling factor for the GARCH adjustment, if needed
        alpha = alpha

        # GARCH forecasts for error
        garch_forecast_error = alpha * standardized_residuals

        return garch_std_combined, garch_forecast_error
    else:
        print("There are no benefits of using an ARCH / GARCH model")
        return None, residuals


def metrics(y_test, y_pred):
    """
    Calculate and display error metrics for model evaluation.

    This function computes standard error metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), and R-squared (R2), to evaluate the accuracy of model predictions.

    Parameters:
    - y_test (Series or array-like): The true values for the target variable in the test set.
    - y_pred (Series or array-like): The predicted values for the target variable in the test set.

    Returns:
    - None
    """
    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print error metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2):", r2)
    return rmse


def ensemble_model(merged_df_2024, variable, fft_threshold):
    """
    Generate an ensemble forecast by combining Prophet and ARIMA models.

    Parameters:
    - merged_df_2024 (DataFrame): The main DataFrame containing time series data to be used for forecasting.
    - variable (str): The target variable for which predictions are to be generated.
    - fft_threshold (float): The threshold for filtering frequencies in the Fast Fourier Transform (FFT) for Prophet.
    - p (int): The order of the autoregressive (AR) term in the ARIMA model for the residuals.
    - q (int): The order of the moving average (MA) term in the ARIMA model for the residuals.

    Returns:
    - DataFrame: A DataFrame containing the original data, Prophet predictions, ARIMA residual forecasts, 
      and the final combined forecast.

    Steps:
    1. Calculate initial predictions with Prophet:
       - Use Prophet to generate predictions for the target variable, applying FFT with the specified threshold to 
         preprocess the time series data and isolate important frequencies.
       - Set 'ds' as the index for Prophet predictions to align with the datetime index of the main DataFrame.
    
    2. Calculate residuals between actual values and Prophet predictions:
       - Compute residuals by subtracting Prophet’s forecast from the actual values in `variable`.
    
    3. Fit ARIMA on the residuals:
       - Fit an ARIMA model on the residuals, using specified AR and MA orders (`p` and `q`), to capture any 
         remaining patterns not accounted for by Prophet.

    4. Combine Prophet and ARIMA forecasts:
       - Add the ARIMA forecasted residuals back to the initial Prophet predictions to produce the final ensemble 
         forecast.
       - Calculate the final residuals as the difference between actual values and the combined forecast to assess 
         the accuracy of the ensemble model.

    Returns:
    - The updated DataFrame with columns for Prophet's forecast, ARIMA residuals, combined forecast, and final residuals.
    """
    # Step 1: Calculate predictions with Prophet
    preds_SP = prophet_predictions(merged_df_2024, variable, calculate_fft(merged_df_2024, variable, fft_threshold))
    # Merging the predictions of prophet to merged df
    preds_SP = preds_SP.set_index('ds')
    preds_SP.index.name = 'Datetime'

    # Convert both indexes to datetime format
    preds_SP.index = pd.to_datetime(preds_SP.index)
    merged_df_2024.index = pd.to_datetime(merged_df_2024.index)
    # Merge both dataframes for convenience
    merged_df_SP = merged_df_2024.join(preds_SP, how='outer')
    merged_df_SP.reset_index(inplace=True)

    # Step 2: Calculate Residuals
    # Calculate residuals as the difference between actual values and Prophet's forecast
    merged_df_SP['Prophet_residuals'] = merged_df_SP[variable] - merged_df_SP["yhat"]

    # Step 3: Fit ARIMA on the Residuals
    # Using the residuals, fit an ARIMA model
    residuals_forecast_series = a_arima(merged_df_SP['Prophet_residuals'])

    # Step 4: Combine the Predictions of ARIMA and Prophet
    # Add the ARIMA residuals forecast back to the Prophet forecast
    merged_df_SP['combined_forecast'] = merged_df_SP['yhat'] + residuals_forecast_series

    #Step 5: calculate the residuals of the combined model. Also, fit a garch model
    # merged_df_SP['ARIMA_residuals'] = merged_df_SP[variable] - merged_df_SP['combined_forecast']
    # garch_std_combined, garch_forecast_error = garch(merged_df_SP['ARIMA_residuals'], 1.96)
    
    # Step 6 combine the prophet + arima with the garch
    # merged_df_SP['combined_forecast'] = merged_df_SP['yhat'] + residuals_forecast_series + garch_forecast_error
    # merged_df_SP['garch_std_combined'] = garch_std_combined
    
    merged_df_SP = merged_df_SP.drop(columns=["ARIMA_residuals", "Prophet_residuals"], axis=1)
    
    return merged_df_SP[['combined_forecast']]

# Function to find the maximum number of consecutive NaNs filled in a column
# As Angelica Asked
def max_consecutive_nans_filled(df, column):
    """

    This function calculates and returns the maximum number 
    of consecutive NaNs in a column that is to be filled

    """
    # Identify consecutive NaNs
    na_groups = df[column].isna().astype(int).groupby(df[column].notna().cumsum()).sum()
    # Get the maximum number of consecutive NaNs that would be interpolated
    max_consecutive_nans = na_groups.max()
    nans_before = df[column].isna().sum()

    print(f"NaNs in {column}: {nans_before}")
    print(f"Max consecutive NaNs filled for '{column}': {max_consecutive_nans}")
    return


def fill_missing_with_prophet(df, column_name, time_column="Datetime"):
    """
    Use Prophet to fill missing values in a specific column of a DataFrame.
    """
    # Prepare the data for Prophet
    temp_df = df[[time_column, column_name]].rename(columns={time_column: 'ds', column_name: 'y'})

    # Separate known and missing data
    known_data = temp_df.dropna()

    # Initialize and fit the Prophet model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(known_data)

    # Make predictions for the full range of dates in the original data
    future = pd.DataFrame({'ds': temp_df['ds']})
    forecast = model.predict(future)

    # Fill missing values with Prophet predictions
    temp_df.set_index('ds', inplace=True)
    temp_df['yhat'] = forecast.set_index('ds')['yhat']
    temp_df[column_name] = temp_df['y'].combine_first(temp_df['yhat'])

    # Update the original DataFrame with the filled values
    df[column_name] = temp_df[column_name].reindex(df[time_column].values).values
    return df

def fill_missing_with_ensemble(df, column_name, time_column="Datetime", fft_threshold=45):
    """
    Use an ensemble of Prophet and ARIMA to fill missing values in a specific column of a DataFrame,
    ensuring that the ARIMA component is aware of missing rows.

    Parameters:
    - df (DataFrame): Input DataFrame containing time series data.
    - column_name (str): Name of the column with missing values to be filled.
    - time_column (str): Name of the column containing the datetime index (default is "Datetime").
    - fft_threshold (float): Threshold for FFT in the ensemble model.

    Returns:
    - DataFrame: Updated DataFrame with missing values filled for the specified column.
    """
    # Prepare the data for the ensemble model
    temp_df = df[[time_column, column_name]].rename(columns={time_column: 'Datetime'})
    temp_df.set_index('Datetime', inplace=True)

    # Separate known and full data
    known_data = temp_df.copy()  # Keep the original structure
    known_data['observed'] = known_data[column_name]  # Store original data

    # Use ensemble_model to predict the full range, including gaps
    forecast_df = ensemble_model(temp_df, column_name, fft_threshold)

    # Combine the forecast with the original data
    known_data['forecast'] = forecast_df['combined_forecast'][:len(known_data)]

    # Fill missing values using the forecast
    known_data[column_name] = known_data['observed'].combine_first(known_data['forecast'])

    # Update the original DataFrame with the filled values
    df[column_name] = known_data[column_name].reindex(df[time_column].values).values

    return df

# Replace the "No Data Available" by 0s in the BSAD columns where applicable
# if all three are missing we just let them be replaced by NaNs

# Replace "No Data Available" in "BSAD_Turn_Up" with 0 if "BSAD_Total" is equal to other column
merged_df.loc[(merged_df["BSAD_Turn_Up"].isna()) & (merged_df["BSAD_Total"] == merged_df["BSAD_Turn_Down"]), "BSAD_Turn_Up"] = 0

# Replace "No Data Available" in "BSAD_Turn_Down" with 0 if "BSAD_Total" is equal to other column
merged_df.loc[(merged_df["BSAD_Turn_Down"].isna()) & (merged_df["BSAD_Total"] == merged_df["BSAD_Turn_Up"]), "BSAD_Turn_Down"] = 0    

# Replace 'NIV_Outturn' with NaN if both 'BM_Bid_Acceptances' and 'BM_Offer_Acceptances' are NaN and 'NIV_Outturn' is 0
merged_df.loc[(merged_df['NIV_Outturn'] == 0) & merged_df['BM_Bid_Acceptances'].isna() & merged_df['BM_Offer_Acceptances'].isna(), 'NIV_Outturn'] = np.nan

# Replace 'NIV_Outturn' with the negative of the sum of 'BM_Offer_Acceptances' and 'BM_Bid_Acceptances' 
# if 'NIV_Outturn' is zero and neither of the other two columns contains NaN
merged_df.loc[(merged_df['NIV_Outturn'] == 0) & merged_df['BM_Offer_Acceptances'].notna() & merged_df['BM_Bid_Acceptances'].notna(), 'NIV_Outturn'] = -(merged_df['BM_Offer_Acceptances'] + merged_df['BM_Bid_Acceptances'])

# Extrapolate 'BM_Bid_Acceptances' with condition to set both columns to NaN if bid check fails
bid_values = -merged_df['NIV_Outturn'] - merged_df['BM_Offer_Acceptances']
merged_df.loc[merged_df['BM_Bid_Acceptances'].isna() & merged_df['NIV_Outturn'].notna(), 'BM_Bid_Acceptances'] = bid_values.where(bid_values <= 0)
merged_df.loc[merged_df['BM_Bid_Acceptances'].isna(), 'BM_Offer_Acceptances'] = np.nan

# Extrapolate 'BM_Offer_Acceptances' with condition to set both columns to NaN if offer check fails
offer_values = -merged_df['NIV_Outturn'] - merged_df['BM_Bid_Acceptances']
merged_df.loc[merged_df['BM_Offer_Acceptances'].isna() & merged_df['NIV_Outturn'].notna(), 'BM_Offer_Acceptances'] = offer_values.where(offer_values >= 0)
merged_df.loc[merged_df['BM_Offer_Acceptances'].isna(), 'BM_Bid_Acceptances'] = np.nan

for column in merged_df.columns:
    merged_df[f'{column}_missing'] = merged_df[column].isnull().astype(int)
    


# Define the function to process each column
def process_column(column):
    if column != 'Datetime':  # Skip the time column
        # Check if there are missing values in the column
        if merged_df[column].isna().sum() != 0:
            # Print information about NaNs
            max_consecutive_nans_filled(merged_df, column)
            # Fill missing values using Prophet
            filled_df = fill_missing_with_ensemble(merged_df, column)
            print(f"Missing values in '{column}' have been filled using Prophet.\n")
            return column, filled_df[column]
    return None

# Parallelize using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_column, merged_df.columns))

# Combine the results back into the DataFrame
for result in results:
    if result:  # Ensure result is not None
        column, filled_data = result
        merged_df[column] = filled_data

# Display updated DataFrame
print("All missing values have been filled.")
# In order to save the merged to csv for easier retrieval
merged_df.to_csv("merged_df_ensemble_filled.csv", index=False)