### Main Script
import scripts.data_extraction as de
import scripts.data_preprocessing as dp
import scripts.model_training as mt


def main():
    # Step 1: Data Extraction
    print("Extracting data...")
    wdi_gdp_df = de.fetch_wdi_data('NY.GDP.MKTP.CD', 2000, 2022)
    de.save_data(wdi_gdp_df, 'data/raw/wdi_gdp.csv')
    imf_df = de.fetch_imf_dot_data('USA', 2000, 2022)
    de.save_data(imf_df, 'data/raw/imf_dot.csv')

    # Extract additional human development indicators
    indicators = {
        'life_expectancy': 'SP.DYN.LE00.IN',
        'primary_enrollment': 'SE.PRM.ENRR',
        'literacy_rate': 'SE.ADT.LITR.ZS',
        'electricity_access': 'EG.ELC.ACCS.ZS',
        'poverty_headcount': 'SI.POV.DDAY'
    }

    for key, indicator in indicators.items():
        df = de.fetch_wdi_data(indicator, 2000, 2022)
        de.save_data(df, f'data/raw/{key}.csv')

    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    wdi_gdp_df = dp.preprocess_wdi_data('data/raw/wdi_gdp.csv')
    dp.save_preprocessed_data(wdi_gdp_df, 'data/processed/wdi_gdp.csv')
    imf_df = dp.preprocess_imf_data('data/raw/imf_dot.csv')
    dp.save_preprocessed_data(imf_df, 'data/processed/imf_dot.csv')

    for indicator in indicators.keys():
        df = dp.preprocess_wdi_data(f'data/raw/{indicator}.csv')
        df = dp.normalize_data(df, ['value'])
        dp.save_preprocessed_data(df, f'data/processed/{indicator}.csv')

    # Step 3: Model Training
    print("Training model...")
    paths = {
        'gdp': 'data/processed/wdi_gdp.csv',
        'life_expectancy': 'data/processed/life_expectancy.csv',
        'primary_enrollment': 'data/processed/primary_enrollment.csv',
        'literacy_rate': 'data/processed/literacy_rate.csv',
        'electricity_access': 'data/processed/electricity_access.csv',
        'poverty_headcount': 'data/processed/poverty_headcount.csv'
    }

    merged_df = mt.load_data(paths)
    X = merged_df[
        ['year', 'value_life_expectancy', 'value_primary_enrollment', 'value_literacy_rate', 'value_electricity_access',
         'value_poverty_headcount']]
    y = merged_df['value_gdp']
    model, mse = mt.train_model(X, y)
    print(f'Model Mean Squared Error: {mse}')


if __name__ == "__main__":
    main()
