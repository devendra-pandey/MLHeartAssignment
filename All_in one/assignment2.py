import requests
from requests.exceptions import Timeout 

url = 'https://github.com/'
try:
    
    r = requests.get(url ,timeout=1)

    print("## status code ###")
    print(r.status_code)

    print("## Directory method to find all method available ###")
    print(dir(r))

    print("## Content ###")
    print(r.content)

    print("## text ###")
    print(r.text)

    print("## JSon")
    print(r.json)

    print("##Headers ###")
    print(r.headers)

except:
    print("Your time limit is too low to get the data")