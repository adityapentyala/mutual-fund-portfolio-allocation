import requests
import pandas as pd
import time

def fetch_nav_history(scheme_code, years=10, frequency='D'):
    """
    Fetches historical NAV data using AMFI scheme codes.
    frequency: 'D' for Daily (default), 'W' for Weekly (Fridays)
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for scheme code {scheme_code}")
        
    data = response.json()
    if not data.get("data"):
        raise ValueError(f"No NAV data available for {scheme_code}")
        
    # Convert JSON payload to a pandas DataFrame
    _df = pd.DataFrame(data['data'])
    df = pd.DataFrame()
    # Parse dates and cast NAV as float
    df['date'] = pd.to_datetime(_df['date'], format='%d-%m-%Y')
    df[f'{scheme_code}_nav'] = pd.to_numeric(_df['nav'])
    
    # Sort chronologically and set date as the DataFrame index
    df = df.sort_values('date').set_index('date')
    
    # Filter dataset for the trailing N years
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff_date]
    
    # Resample frequency
    if frequency.upper() == 'W':
        # 'W-FRI' resamples the data to weeks ending on Friday.
        # .last() pulls the Friday NAV, or Thursday if Friday was a market holiday.
        df = df.resample('W-FRI').last()
        
    return df

def build_portfolio_matrix(codes, years=5, frequency='W'):
    portfolio_series = []
    
    for code in codes:
        print(f"Processing: {code}...")
        
        if code:
            try:
                # Fetch history and extract just the 'nav' column as a Series
                df_history = fetch_nav_history(code, years=years, frequency=frequency)
                
                # Rename the series to the fund name for column identification
                nav_series = df_history[f'{code}_nav'].rename(code)
                portfolio_series.append(nav_series)
                
            except Exception as e:
                print(f"  -> Error fetching data for {code}: {e}")
        else:
            print(f"  -> AMFI code not found for {code}")
            
        # Respect AMFI API rate limits
        time.sleep(0.5)
        
    # Combine all individual series into one DataFrame aligned by the date index
    if portfolio_series:
        master_df = pd.concat(portfolio_series, axis=1).sort_index()
        # Forward-fill missing NAVs (due to varying market holidays) then drop NaNs
        clean_matrix = master_df.ffill()
        return clean_matrix
    else:
        return pd.DataFrame()