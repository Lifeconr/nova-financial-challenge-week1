import pandas as pd

# Paths
input_path = r"C:\Users\Senayit\Documents\1\week1\nova-financial-challenge-week1\Data\raw_analyst_ratings.csv"
output_path = r"C:\Users\Senayit\Documents\1\week1\nova-financial-challenge-week1\Data\corrected_analyst_ratings.csv"
invalid_output_path = r"C:\Users\Senayit\Documents\1\week1\nova-financial-challenge-week1\Data\invalid_dates.csv"

# Load the dataset
df = pd.read_csv(input_path)
print(f"Initial row count: {len(df)}")
print("Sample of original dates:", df['date'].head(10).to_list())

# Function to parse dates with multiple formats
def parse_date(date_str):
    if pd.isna(date_str) or not isinstance(date_str, str) or date_str.strip() == '':
        return pd.NaT
    # List of formats to try
    formats = [
        '%Y-%m-%d %H:%M:%S%z',  # ISO 8601 with timezone (e.g., 2020-06-05 10:30:54-04:00)
        '%m/%d/%Y %H:%M',       # US format with time (e.g., 5/14/2013 0:00)
        '%m/%d/%Y',             # US format without time (e.g., 5/14/2013)
        '%Y-%m-%d',             # YYYY-MM-DD (e.g., 2020-05-22)
        '%Y/%m/%d %H:%M:%S%z',  # Alternative ISO-like with slashes
        '%Y-%m-%d %H:%M:%S',    # YYYY-MM-DD with time, no timezone
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt, errors='raise')
        except ValueError:
            continue
    # Fallback to infer_datetime_format if no format matches
    try:
        return pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')
    except ValueError:
        return pd.NaT

# Apply parsing
original_len = len(df)
df['date'] = df['date'].apply(parse_date)
print("After parsing, date column type:", df['date'].dtype)
print("Sample of parsed dates:", df['date'].head(10).to_list())

# Identify and save invalid dates
invalid_dates = df[df['date'].isnull()]
if not invalid_dates.empty:
    print(f"Found {len(invalid_dates)} rows with unparseable dates. Saving to {invalid_output_path}.")
    # Save the original date values for inspection
    invalid_dates_with_original = invalid_dates.copy()
    invalid_dates_with_original['original_date'] = invalid_dates_with_original.index.map(lambda x: df['date'][x] if x < len(df) else None)
    print("Sample of unparseable dates (original values):", invalid_dates_with_original['original_date'].head().to_list())
    invalid_dates_with_original.to_csv(invalid_output_path, index=False)
    df = df.dropna(subset=['date'])
    print(f"Dropped {original_len - len(df)} rows. Remaining rows: {len(df)}")

# Ensure the date column is a datetime Series
df['date'] = pd.to_datetime(df['date'], errors='coerce')
if df['date'].isnull().any():
    print("Warning: Some dates are still NaT after conversion. These will be dropped.")
    df = df.dropna(subset=['date'])
    print(f"Dropped additional {original_len - len(df)} rows due to NaT values. Remaining rows: {len(df)}")

# Standardize timezone: if no timezone, assume UTC-4; then convert to UTC
df['date'] = df['date'].apply(lambda x: x.tz_localize('Etc/GMT+4') if x.tzinfo is None else x)
df['date'] = df['date'].dt.tz_convert('UTC')
print("Sample of dates after timezone conversion:", df['date'].head(10).to_list())

# Save the corrected dataset
df.to_csv(output_path, index=False)
print(f"Corrected dataset saved to {output_path}")