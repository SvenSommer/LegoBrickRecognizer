import os
import numpy as np
import json
import requests
from timeit import default_timer as time


class GoogleDriveUploader:
    def __init__(self, access_token, refresh_token):
        self.token = {
            "access_token": access_token,
            "scope": "https://www.googleapis.com/auth/drive",
            "token_type": "Bearer",
            "expires_in": 3599,
            "refresh_token": refresh_token
        }

    def uploadFile(self, filepath, folder_id):
        if not os.path.isfile(filepath):
            print("Error: File to upload '{}' not found.".format(filepath))
            return

        headers = {"Authorization": 'Bearer {}'.format(self.token['access_token'])}
        para = {
            "name": os.path.basename(filepath),
            'parents': [folder_id]
        }
        files = {
            'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
            'file': open(filepath, "rb")

        }

        start_time = time()
        r = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            headers=headers,
            files=files
        )
        finish_time = time()
        print('Time: {:.5f} sec'.format(finish_time - start_time))
        print(r.text)
