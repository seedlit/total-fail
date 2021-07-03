import pandas as pd
from datetime import datetime, timedelta
import calendar
import os
import random


def get_random_datetime(min_year, max_year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()

if __name__ == "__main__":   


    csv_path = "elonmusk.csv"
    out_dir = "elon_features"

    os.makedirs(out_dir, exist_ok=True)    
    target_years_list = [2020, 2021]
    weekday_list = []
    hour_list = []
    minute_list = []
    month_list = []
    year_list = []
    date_list = []
    tweet_bool_list = []

    for target_year in target_years_list:
        csv_pd = pd.read_csv(csv_path)
        datetime_list = csv_pd["date"].to_list()        
        for i in datetime_list:
            datetime_obj = datetime.fromisoformat(i)            
            if datetime_obj.year == target_year:                
                # weekday_list.append(calendar.day_name[datetime_obj.weekday()])
                weekday_list.append(datetime_obj.weekday())  # 0 --> Monday; 6 --> Sunday
                hour_list.append(datetime_obj.hour)
                minute_list.append(datetime_obj.minute)
                month_list.append(datetime_obj.month)
                year_list.append(datetime_obj.year)
                date_list.append(datetime_obj.day)
                tweet_bool_list.append(1)
                # adding a random negative example
                random_datetime_obj = get_random_datetime(2020, 2021)
                if random_datetime_obj != datetime_obj:
                    weekday_list.append(random_datetime_obj.weekday())  # 0 --> Monday; 6 --> Sunday
                    hour_list.append(random_datetime_obj.hour)
                    minute_list.append(random_datetime_obj.minute)
                    month_list.append(random_datetime_obj.month)
                    year_list.append(random_datetime_obj.year)
                    date_list.append(random_datetime_obj.day)
                    tweet_bool_list.append(0)


    # converting to a pandas dataframe
    features_data = {
        "year": year_list,
        "month": month_list,
        "date": date_list,
        "hour": hour_list,
        "minute": minute_list,
        "weekday": weekday_list,
        "tweet_bool": tweet_bool_list,
    }
    
    features_df = pd.DataFrame(features_data)
    # saving as csv
    out_csv_path = os.path.join(out_dir, csv_path.replace(".csv", "_features.csv"))
    features_df.to_csv(out_csv_path)
