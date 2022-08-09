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


# check if we can access zenodo
print(f"Accessing zenodo with token: {ACCESS_TOKEN}")
r = requests.post(
        ZENODO_URL,
        params={'access_token': ACCESS_TOKEN}, json={},
)
print(f"Access to zendo: {r.json()}")


bucket_url = r.json()['links']['bucket']
print(f"Uploading {FILEPATH}")
r = requests.put(
        '%s/%s' % (bucket_url,FILEPATH),
        data=open(FILEPATH, 'rb'),
        headers={"Accept":"application/json",
        "Authorization":"Bearer %s" % ACCESS_TOKEN,
        "Content-Type":"application/octet-stream"}
)


print(f"Upload status: {r.json()}")
