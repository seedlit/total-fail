import snscrape.modules
import csv
import os


if __name__ == "__main__":

    twitter_username = "sujayvkumar"
    out_dir = "testing2"

    os.makedirs(out_dir, exist_ok=True)
    csv_name = "{}.csv".format(twitter_username)

    tweets = snscrape.modules.twitter.TwitterUserScraper(username=twitter_username)    

    data_list = []
    line_count = 1
    for i in tweets.get_items():        
        if line_count == 1:
            field_names = list(i._fields)
            line_count += 1
        temp_dict = {}
        for field in field_names:
            temp_dict[field] = getattr(i, field)
        data_list.append(temp_dict)
        
    with open(os.path.join(out_dir, csv_name), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data_list)