# Upload big files to Zenodo.
#
# usage: ./zenodo_upload.py [filename]
#

import requests
import sys
import os

ACCESS_TOKEN=os.environ['SANDBOX_TOKEN']
DEPOSITION='1091256'
FILEPATH=sys.argv[1]
ZENODO_URL='https://sandbox.zenodo.org/api/deposit/depositions'


print(f"Accessing zenodo with token: {ACCESS_TOKEN}")
print(f"Uploading {FILEPATH}")
r = requests.put(
        f"https://zenodo.org/api/deposit/depositions/{DEPOSITION}/files",
        data=open(FILEPATH, 'rb'),
        headers={"Accept":"application/json",
        f"Authorization":f"Bearer {ACCESS_TOKEN}",
        "Content-Type":"application/octet-stream"}
)
print(f"Upload status:\n{r.json()}")
