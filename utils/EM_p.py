import json

import requests

group_id = '1913402932208866274'  # Type your group id
api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'

#Audio file upload
url = f'https://api.minimaxi.chat/v1/files/upload?GroupId={group_id}'
headers1 = {
    'authority': 'api.minimaxi.chat',
    'Authorization': f'Bearer {api_key}'
}

data = {
    'purpose': 'voice_clone'
}

files = {
    'file': open('i.wav', 'rb')
}
response = requests.post(url, headers=headers1, data=data, files=files)
try:
    resp_json = response.json()
    print("文件上传返回：", resp_json)
    file_info = resp_json.get("file")
    if file_info is None:
        print("接口返回没有 file 字段，返回内容：", resp_json)
        file_id = None
    else:
        file_id = file_info.get("file_id")
        print("file_id:", file_id)
except Exception as e:
    print("解析文件上传返回时出错：", e)
    print("原始返回内容：", response.text)
    file_id = None

#Voice cloning
url = f'https://api.minimaxi.chat/v1/voice_clone?GroupId={group_id}'
payload2 = json.dumps({
  "file_id": file_id,
  "voice_id": "ppooiudiii"
})
headers2 = {
  'authorization': f'Bearer {api_key}',
  'content-type': 'application/json'
}
response = requests.request("POST", url, headers=headers2, data=payload2)
print(response.text)



