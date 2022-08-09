# Upload big files to Zenodo.
#
# usage: ./zenodo_upload.py [filename]
#

import requests
import sys
imoport os

ACCESS_TOKEN=os.environ['ZENODO_TOKEN']
DEPOSITION='1091256'
FILEPATH=sys.argv[1]
ZENODO_URL='https://sandbox.zenodo.org/api/deposit/depositions'


# check if we can access zenodo
r = requests.post(
        ZENODO_URL,
        params={'access_token': ACCESS_TOKEN}, json={},
        headers={"Content-Type": "application/json"}
)
print(f"Access to zendo: {r.status_code}")


bucket_url = r.json()['links']['bucket']
print(f"Uploading {}")
r = requests.put(
        '%s/%s' % (bucket_url,filename),
        data=open(FILEPATH, 'rb'),
        headers={"Accept":"application/json",
        "Authorization":"Bearer %s" % ACCESS_TOKEN,
        "Content-Type":"application/octet-stream"}
)


print(r.status_code)
