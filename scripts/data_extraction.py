import pandas as pd
import requests


def fetch_wdi_data(indicator, start_year, end_year):
    """
    Fetch WDI data for a given indicator and date range.

    Parameters:
    - indicator: The indicator code (e.g., 'NY.GDP.MKTP.CD' for GDP).
    - start_year: The start year for the data range.
    - end_year: The end year for the data range.

    Returns:
    - df: A pandas DataFrame containing the fetched data.
    """
    url = f'http://api.worldbank.org/v2/country/all/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=20000'
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data[1])
    return df


def fetch_imf_dot_data(country, start_year, end_year):
    """
    Fetch IMF DOT data for a given country and date range.

    Parameters:
    - country: The country code (e.g., 'USA').
    - start_year: The start year for the data range.
    - end_year: The end year for the data range.

    Returns:
    - df: A pandas DataFrame containing the fetched data.
    """
    # Placeholder for actual IMF DOT data extraction
    url = f'http://imf.api.example.com/data/{country}/{start_year}/{end_year}'
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data)
    return df


def save_data(df, path):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    - df: The DataFrame to save.
    - path: The file path to save the CSV.
    """
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Extracting GDP data
    wdi_gdp_df = fetch_wdi_data('NY.GDP.MKTP.CD', 2000, 2022)
    save_data(wdi_gdp_df, 'data/raw/wdi_gdp.csv')

    # Extracting IMF DOT data
    imf_df = fetch_imf_dot_data('USA', 2000, 2022)
    save_data(imf_df, 'data/raw/imf_dot.csv')

    # Extracting additional human development indicators
    indicators = {
        'life_expectancy': 'SP.DYN.LE00.IN',
        'primary_enrollment': 'SE.PRM.ENRR',
        'literacy_rate': 'SE.ADT.LITR.ZS',
        'electricity_access': 'EG.ELC.ACCS.ZS',
        'poverty_headcount': 'SI.POV.DDAY'
    }

    for key, indicator in indicators.items():
        df = fetch_wdi_data(indicator, 2000, 2022)
        save_data(df, f'data/raw/{key}.csv')
