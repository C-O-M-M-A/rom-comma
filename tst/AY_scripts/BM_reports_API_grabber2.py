"""This script repeatedly queries the BMReports API and compiles the results.
within date and settlement period loops. The example is wind generator output.
Other APIs may require different treatment, according to the docs at:
https://github.com/C-O-M-M-A/Electrical_Grid_System_Analysis/blob/master/
bmrs_api_data_push_user_guide_v1.1.pdf
Some additional formatting of the output will be requried."""

import httplib2
from pprint import pformat


def post_elexon(url):
    http_obj = httplib2.Http()
    resp, content = http_obj.request(uri=url, method='GET', headers={'Content-Type': 'application/xml; charset=UTF-8'},)
    print('===Response===')
    print(pformat(resp))
    print('===Content===')
    print(pformat(content))
    print('===Finished===')


def main():
    post_elexon(url='https://api.bmreports.com/BMRS/B1770/v1?APIKey=jfnk8kohx7592z1&SettlementDate=2015-03-01&Period=1&ServiceType=xml',)



if __name__ == "__main__":
    main()
