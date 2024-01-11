import base64
import requests

import sys


url = sys.argv[1]

with open(sys.argv[2], "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

payload = {"image_base64": encoded_string}
resp = requests.post(url=url, data=payload)

img_str = resp.json()["image_base64"]

img_recovered = base64.b64decode(img_str)

with open("returned_img.png", "wb") as f:
    f.write(img_recovered)
