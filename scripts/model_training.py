import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data(paths):
    """
    Load data from multiple CSV files and merge them on common columns.

    Parameters:
    - paths: A dictionary with indicator names as keys and file paths as values.

    Returns:
    - merged_df: A merged pandas DataFrame containing all indicators.
    """
    dataframes = {key: pd.read_csv(path) for key, path in paths.items()}
    merged_df = pd.merge(dataframes['gdp'], dataframes['life_expectancy'], on=['country', 'year'],
                         suffixes=('_gdp', '_life_expectancy'))
    merged_df = pd.merge(merged_df, dataframes['primary_enrollment'], on=['country', 'year'],
                         suffixes=('', '_primary_enrollment'))
    merged_df = pd.merge(merged_df, dataframes['literacy_rate'], on=['country', 'year'],
                         suffixes=('', '_literacy_rate'))
    merged_df = pd.merge(merged_df, dataframes['electricity_access'], on=['country', 'year'],
                         suffixes=('', '_electricity_access'))
    merged_df = pd.merge(merged_df, dataframes['poverty_headcount'], on=['country', 'year'],
                         suffixes=('', '_poverty_headcount'))
    return merged_df


def train_model(X, y):
    """
    Train a linear regression model and evaluate its performance.

    Parameters:
    - X: Features for training.
    - y: Target variable for training.

    Returns:
    - model: The trained linear regression model.
    - mse: Mean Squared Error of the model on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


if __name__ == "__main__":
    # Define paths to the preprocessed data files
    paths = {
        'gdp': 'data/processed/wdi_gdp.csv',
        'life_expectancy': 'data/processed/life_expectancy.csv',
        'primary_enrollment': 'data/processed/primary_enrollment.csv',
        'literacy_rate': 'data/processed/literacy_rate.csv',
        'electricity_access': 'data/processed/electricity_access.csv',
        'poverty_headcount': 'data/processed/poverty_headcount.csv'
    }

    # Load and merge data
    merged_df = load_data(paths)

    # Prepare features and target variable
    X = merged_df[
        ['year', 'value_life_expectancy', 'value_primary_enrollment', 'value_literacy_rate', 'value_electricity_access',
         'value_poverty_headcount']]
    y = merged_df['value_gdp']

    # Train model and evaluate performance
    model, mse = train_model(X, y)
    print(f'Model Mean Squared Error: {mse}')
