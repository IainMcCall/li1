"""
Provides functions to extract equity calendar data (Earnings annoucements, dividends).
"""
import finnhub
import pandas as pd

import CONFIG


def get_earnings_calendar_data(start_date, end_date):
    """
    Get updated Inflation rates for only required dates.

    Args:
        start_date (date): Date from which to extract earnings calendar data.
        end_date (date): Date to which to extract calendar data.
    Returns:
        (pandas.core.Frame.DataFrame): DataFrame containing historical
    """
    finnhub_client = finnhub.Client(api_key=CONFIG.FINNHUB_API_KEY)
    data = finnhub_client.earnings_calendar(_from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'),
                                            symbol="", international=False) # To add international from other source.
    earnings_calendars = pd.DataFrame()
    for d in data['earningsCalendar']:
        t = d['symbol']
        year = d['year']
        quarter = d['quarter']
        name = f"{t}_{year}_{quarter}"
        for i in d:
            earnings_calendars.at[name, i] = d[i]
    return earnings_calendars
