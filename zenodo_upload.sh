#!/bin/bash
# Upload big files to Zenodo.
#
# usage: ./zenodo_upload.sh [filename]
#

set -e

DEPOSITION='1091256'
FILEPATH="$1"
FILENAME=$(echo $FILEPATH | sed 's+.*/++g')

BUCKET=$(curl https://sandbox.zenodo.org/api/deposit/depositions/"$DEPOSITION"?access_token="$ZENODO_TOKEN" | jq --raw-output .links.bucket)

curl --progress-bar -o /dev/null --upload-file "$FILEPATH" $BUCKET/"$FILENAME"?access_token="$ZENODO_TOKEN"
