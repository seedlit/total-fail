import pandas as pd
from datetime import datetime
import calendar
import os


if __name__ == "__main__":

    csv_path = "elonmusk.csv"
    out_dir = "elon_plots"
    target_years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

    for target_year in target_years_list:

        os.makedirs(out_dir, exist_ok=True)
        csv_pd = pd.read_csv(csv_path)
        datetime_column = csv_pd['date']

        weekday_count_dict = {"Monday":0, "Tuesday":0, "Wednesday": 0, "Thursday": 0, "Friday":0, "Saturday": 0, "Sunday": 0}
        for i in range(len(datetime_column)):        
            datetime_obj = datetime.fromisoformat(datetime_column.iloc[i])
            if datetime_obj.year == target_year:
                weekday_count_dict[calendar.day_name[datetime_obj.weekday()]] += 1

        print(weekday_count_dict)

        import matplotlib.pyplot as plt
        D = weekday_count_dict
        plt.bar(range(len(D)), list(D.values()), align='center')
        plt.xticks(range(len(D)), list(D.keys()))
        plt.savefig(os.path.join(out_dir, "year_{}.png".format(target_year)))