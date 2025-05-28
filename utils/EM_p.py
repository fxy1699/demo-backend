import json

import requests

group_id = '1914687288663609686'  # Type your group id
api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyBXYW5nIiwiVXNlck5hbWUiOiJTaGFuZyBXYW5nIiwiQWNjb3VudCI6IiIsIlN1YmplY3RJRCI6IjE5MTQ2ODcyODg2Njc4MDM5OTAiLCJQaG9uZSI6IiIsIkdyb3VwSUQiOiIxOTE0Njg3Mjg4NjYzNjA5Njg2IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiMjQ2ODM1OTU1OEBxcS5jb20iLCJDcmVhdGVUaW1lIjoiMjAyNS0wNC0yMyAwMDoyNjo0MCIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.ovfCL3yV07JQ1uadDhjVO6bdOlV1Y_kgCEaNA38cFrrPQa1wndNfpKsD7fYJH260e1GuvODcboX--ahd8XOYFvCTsaefrFAuZuuGKkRO_V6E9AkqfTMM4tVR0CHsioqzb3HXzv4EeJusq-LZAwD2uRs6AJ8OwRE6GkSRrMnbNYuk2RIfk8o4jMGTidZO9thPVuIlOO2yQIMP2bGzpSbaLdKt_8dZw25zyfS0t3bOozWAqcVbxhbDeDGl8sfi6Fm4hjXtKVd6-jUhdEGn6PQ4GraihkGiWde6Xj_7qLfa7frxksSgypILQl1dJl91vfxW-SwRD8UW_Ox8g9gFoKyuow'  # Type your api key

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
    'file': open('i.m4a', 'rb')
}
response = requests.post(url, headers=headers1, data=data, files=files)
file_id = response.json().get("file").get("file_id")
print(file_id)

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



