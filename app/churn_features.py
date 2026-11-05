def add_engineered_features(df):
    """Adds engineered features used by the churn model.
    Must be called on any new data before passing it to the model pipeline."""
    df = df.copy()
    df['charge_per_tenure_month'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    return df
