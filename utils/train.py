# This Python file uses the following encoding: utf-8

import json
import subprocess
import time
from typing import Iterator

import requests

group_id = '1913402932208866274'    #your_group_id
api_key = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGFuZyIsIlVzZXJOYW1lIjoiU2hhbmciLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTkxMzQwMjkzMjIxMzA2MDU3OCIsIlBob25lIjoiMTg4NTE2NzU0ODciLCJHcm91cElEIjoiMTkxMzQwMjkzMjIwODg2NjI3NCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI1LTA1LTI5IDEyOjI0OjI1IiwiVG9rZW5UeXBlIjoxLCJpc3MiOiJtaW5pbWF4In0.Hvrq5qoWInpOshgsrGBWqD2MNZ41JKC7Cx-PUlpDxi5UQbYmN1uCgDj36CANoZK8v9-nQjoLb5jdPywH6J_P6H94uQluhEf-v0nWBa1NFCL5F5eaHYGjiUCyisl9o7qBcbgqJsKiCZkOXKZs8MnLjiptnQb1NxliPIs-7jflUNPvsELfWt8y3-dJGFayfDnvYvRwnpPyqn9rb7h3Qr18aiQ3jcND-SXFfou11hLBL5gvf9h5Ci1hhvKrWlOyVHQ8y2z3KlcfjR5umn4gI2Bcrr-XPYUl1xnOsSw0vKTivjpWcJCdfy5bJ0w-ZZI1T3wyhbsc2H3d26xy_HU_WjN0-w'

file_format = 'mp3'  # support mp3/pcm/flac

url = "https://api.minimaxi.chat/v1/t2a_v2?GroupId=" + group_id
headers = {"Content-Type":"application/json", "Authorization":"Bearer " + api_key}


def build_tts_stream_headers() -> dict:
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + api_key,
    }
    return headers


def build_tts_stream_body(text: str) -> dict:
    body = json.dumps({
        "model":"speech-02-turbo",
        "text": text,
        "stream":True,
        "voice_setting":{
            "voice_id":"ppooiudiii",
            "speed":1.0,
            "vol":1.0,
            "pitch":0
        },
        "audio_setting":{
            "sample_rate":32000,
            "bitrate":128000,
            "format":"mp3",
            "channel":1
        }
    })
    return body

mpv_exe_path = r"D:\mpv-x86_64-20250420-git-3600c71\mpv.exe"
mpv_command = [mpv_exe_path, "--no-cache", "--no-terminal", "--", "fd://0"]
mpv_process = subprocess.Popen(
    mpv_command,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


def call_tts_stream(text: str) -> Iterator[bytes]:
    tts_url = url
    tts_headers = build_tts_stream_headers()
    tts_body = build_tts_stream_body(text)

    response = requests.request("POST", tts_url, stream=True, headers=tts_headers, data=tts_body)
    for chunk in (response.raw):
        if chunk:
            if chunk[:5] == b'data:':
                data = json.loads(chunk[5:])
                if "data" in data and "extra_info" not in data:
                    if "audio" in data["data"]:
                        audio = data["data"]['audio']
                        yield audio


def audio_play(audio_stream: Iterator[bytes]) -> bytes:
    audio = b""
    for chunk in audio_stream:
        if chunk is not None and chunk != '\n':
            decoded_hex = bytes.fromhex(chunk)
            mpv_process.stdin.write(decoded_hex)  # type: ignore
            mpv_process.stdin.flush()
            audio += decoded_hex

    return audio

text_to_speak = "很高兴认识你！" 
audio_chunk_iterator = call_tts_stream(text_to_speak)
audio = audio_play(audio_chunk_iterator)

# save results to file
timestamp = int(time.time())
file_name = f'output_total_{timestamp}.{file_format}'
with open(file_name, 'wb') as file:
    file.write(audio)