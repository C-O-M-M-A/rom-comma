"""This script repeatedly queries the BMReports API and compiles the results.
within date and settlement period loops. The example is wind generator output.
Other APIs may require different treatment, according to the docs at:
https://github.com/C-O-M-M-A/Electrical_Grid_System_Analysis/blob/master/
bmrs_api_data_push_user_guide_v1.1.pdf
Some additional formatting of the output will be requried."""
import requests
import datetime
import pandas as pd

# The code below creates a list of 2017 dates in format yyyy-mm-dd
start = datetime.date(2017, 1, 1)

dates = []
for i in range(2):
    date = start + datetime.timedelta(i)
    dates = dates + [date]

# the code below queries the API
# api_stem = "https://api.bmreports.com/BMRS/B1610/v2?APIKey=xkuljc1he2a5sho"
api_stem = "https://api.bmreports.com/BMRS/B0620/v2?APIKey=xkuljc1he2a5sho"
# &SettlementDate=<SettlementDate>&Period=<Period>&ServiceType=<xml/csv>
output_table = []
for j in dates:
    for k in range(1, 49):
        response = requests.get(api_stem +
                                "&SettlementDate=" + str(j) +
                                "&Period=" + str(k) +
                                "&ServiceType=csv",    # csv also an option
                                auth=('droberts1@sheffield.ac.uk', 'i8theElectric'))
        print(response.content)
        output_table = output_table + [response.content]
        print(j, k)
    output_table = pd.DataFrame(output_table)

    output_table.to_csv('day_ahead_load.csv', sep=',')