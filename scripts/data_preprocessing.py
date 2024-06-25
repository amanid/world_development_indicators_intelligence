import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_wdi_data(path):
    """
    Preprocess WDI data by dropping NA values and extracting the year.

    Parameters:
    - path: The file path to the raw data CSV.

    Returns:
    - df: A preprocessed pandas DataFrame.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['value'], inplace=True)
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df


def preprocess_imf_data(path):
    """
    Preprocess IMF data by dropping NA values and extracting the year.

    Parameters:
    - path: The file path to the raw data CSV.

    Returns:
    - df: A preprocessed pandas DataFrame.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['trade_value'], inplace=True)
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df


def normalize_data(df, columns):
    """
    Normalize specified columns in the DataFrame using Min-Max Scaling.

    Parameters:
    - df: The DataFrame to normalize.
    - columns: A list of column names to normalize.

    Returns:
    - df: The DataFrame with normalized columns.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def save_preprocessed_data(df, path):
    """
    Save a preprocessed DataFrame to a CSV file.

    Parameters:
    - df: The DataFrame to save.
    - path: The file path to save the CSV.
    """
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Preprocessing GDP data
    wdi_gdp_df = preprocess_wdi_data('data/raw/wdi_gdp.csv')
    save_preprocessed_data(wdi_gdp_df, 'data/processed/wdi_gdp.csv')

    # Preprocessing IMF DOT data
    imf_df = preprocess_imf_data('data/raw/imf_dot.csv')
    save_preprocessed_data(imf_df, 'data/processed/imf_dot.csv')

    # Preprocessing additional human development indicators
    indicators = ['life_expectancy', 'primary_enrollment', 'literacy_rate', 'electricity_access', 'poverty_headcount']

    for indicator in indicators:
        df = preprocess_wdi_data(f'data/raw/{indicator}.csv')
        df = normalize_data(df, ['value'])
        save_preprocessed_data(df, f'data/processed/{indicator}.csv')
