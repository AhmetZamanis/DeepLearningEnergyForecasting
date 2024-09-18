import datetime
import requests
import json
import pandas as pd
import time

from dateutil.relativedelta import relativedelta


def _get_request_dates(years_of_data: int, consumption_lag: int = 2) -> tuple[list, str]:
    """
    Returns the start & end dates for a consumption data request, where
        end date = current date - (consumption_lag) hours.
    
    Returns 1 start date per requested years of data, 
    because the API returns 1 years of data at a time.
        
    Returned dates are in ISO format: '2024-01-30T00:00:00+03:00'
    """

    # Current date, with timezone, without microseconds
    current_date = datetime.datetime.now().astimezone().replace(microsecond = 0)

    # End date as (current date - consumption_lag)
    end_date = current_date - datetime.timedelta(hours = consumption_lag)

    # Loop to get start dates
    start_dates = []
    for year in range(1, years_of_data + 1):
        start_date = end_date - relativedelta(years = year)
        start_dates.append(start_date.isoformat())

    return start_dates, end_date.isoformat()


def _get_consumption_data_1yr(tgt: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Takes a TGT (ticket-granting-ticket), data start date & data end date, maximum 1 years apart.
    Returns daily consumption dataframe with 1 years of data.

    API Docs:
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_adding_security_information_to_requests
    https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html#_realtime-consumption
    """

    # Consumption endpoint
    url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"

    # Headers in docs
    headers = {
                "Accept-Language": "en",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "TGT": tgt,
    }

    # Request date range
    body = {
                "startDate": start_date,
                "endDate": end_date
    }

    # Consumption data request
    response = requests.post(
                url,
                data = json.dumps(body),
                headers = headers
    )
    status_code = response.status_code
    status_bool = response.ok

    # Successful response
    if status_bool == True:

        # Print status code if code unexpected but request successful
        if status_code != 200:
            print(f"Consumption data request response code: {status_code}")

        # Get data
        response_data = json.loads(response.text)
        df = pd.DataFrame(response_data["items"])
        
        return df
        
    raise Exception(f"Consumption data request failed. Status code: {status_code}")


def get_tgt(username: str, password: str) -> str:
    """
    Takes the EPİAŞ API username & password.
    Requests & returns a TGT (ticket-granting-ticket), valid for 2 hours.

    API Docs: 
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_adding_security_information_to_requests
    """

    # TGT endpoint
    url = "https://giris.epias.com.tr/cas/v1/tickets"

    # Headers in docs
    headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/plain"
    }

    # EPİAŞ username & password
    body = {
                "username": username,
                "password": password
    }

    # TGT Request
    response = requests.post(
                url,
                data = body,
                headers = headers
    )
    status_code = response.status_code
    status_bool = response.ok

    # Successful response
    if status_bool == True:

        # Print status code if code unexpected but request successful
        if status_code != 201:
            print(f"TGT request response code: {status_code}")
            
        return response.text
        
    raise Exception(f"TGT request failed. Status code: {status_code}")


def get_consumption_data(tgt:str, years_of_data: int, consumption_lag: int = 2, timeout: int = 5) -> pd.DataFrame:
    """
    Takes a TGT, the number of years requested, and the delay of the realtime consumption data (consumption_lag).
    Returns daily consumption dataframe with 1 or more years of data, ending with (current_date - consumption_lag).

    The API returns 1 year of data per request, so the function makes multiple requests if necessary. 
    `timeout` controls wait time between requests, in seconds.
        
    API Docs:
    https://seffaflik.epias.com.tr/electricity-service/technical/en/index.html#_adding_security_information_to_requests
    https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html#_realtime-consumption
    """

    # Raise value errors for wrong parameter values
    if (years_of_data < 1):
        raise ValueError("Ensure 'years_of_data > 0'.")

    if (consumption_lag < 0):
        raise ValueError("Ensure 'consumption_lag >= 0'.")

    if (timeout < 0):
        raise ValueError("Ensure 'timeout >= 0'.")

    # Get start & end dates
    start_dates, end_date = _get_request_dates(years_of_data, consumption_lag)

    # Loop to get dataframes for each year
    df_list = []
    counter = 0
    for year in range(1, years_of_data):

        # Get data for year
        df_year = _get_consumption_data_1yr(tgt, start_dates[counter], end_date)
        df_list.append(df_year)

        # Shift back end & start dates 1 year
        end_date = start_dates[counter]
        counter += 1

        # Wait
        time.sleep(timeout)

    # Concatenate dataframes, drop duplicates
    df = pd.concat(df_list).drop_duplicates()
    
    return df