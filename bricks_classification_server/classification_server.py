import numpy as np
import os
import sys
from timeit import default_timer as time
from argparse import ArgumentParser, Namespace
from flask import Flask, jsonify, abort, request
import io
import logging
import torch
import torchvision
import json
import time
import requests
import os
from PIL import Image
from threading import Lock


def args_parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=5001)
    return parser.parse_args()


class FunctionServingWrapper(object):
    """
    Class of wrapper for restriction count of simultaneous function calls
    """
    def __init__(self,
                 callable_function: callable,
                 count_of_parallel_users: int = 1):
        self.callable_function = callable_function
        self.resources = [Lock() for _ in range(count_of_parallel_users)]
        self.call_mutex = Lock()

    def __call__(self, *_args, **_kwargs):
        """
        Run call method of target callable function
        Args:
            *_args: args for callable function
            **_kwargs: kwargs for callable function
        Returns:
            Return callable function results
        """
        self.call_mutex.acquire()
        i = -1
        while True:
            for k in range(len(self.resources)):
                if not self.resources[k].locked():
                    i = k
                    break
            if i > -1:
                break

        self.resources[i].acquire()
        self.call_mutex.release()

        result = self.callable_function(*_args, **_kwargs)
        self.resources[i].release()

        return result


class BrickClassification(object):
    def __init__(self,
                 traced_model_file: str,
                 file_names_file: str,
                 device: str = 'cpu'):
        self.model = torch.jit.load(traced_model_file)
        self.model.eval()
        self.model = self.model.to(device)
        self.input_shape = (224, 224)
        self.device = device

        with open(file_names_file, 'r') as f:
            self.classes_names = [line.rstrip() for line in f]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor()
            ]
        )

    def _preprocess(self, _input_img: Image) -> torch.Tensor:
        inp_tensor = self.transform(_input_img).unsqueeze(0).to(self.device)
        return inp_tensor

    def __call__(self, _img: np.ndarray) -> tuple:
        out = self.model(self._preprocess(_img)).detach().to('cpu')[0]
        return self.classes_names[out.argmax()], \
               float(torch.sigmoid(out).max().numpy())


app = Flask(__name__)
app_log = logging.getLogger('werkzeug')
app_log.setLevel(logging.ERROR)

brick_classificator = FunctionServingWrapper(
    BrickClassification(
        traced_model_file='brick_classifier.torchscript.pt',
        file_names_file='names.txt',
        device='cpu'
    )
)


def solution_inference(img: np.ndarray) -> dict:
    """
    Pipeline inference
    Args:
        img: image in uint8 RGB HWC format
    Returns:
        Dictionary with predicted class name
    """
    global brick_classificator

    if img is None:
        abort(409)

    cls_num, conf = brick_classificator(img)

    return {
        'class': cls_num,
        'confidence': float('{:.3f}'.format(conf))
    }


def is_flip(file_path: str) -> bool:
    view_type = file_path.split('_')[-1].split('.')[0]
    return view_type in ['LEFT', 'RIGHT']


def getImageFromUrl(image_url_str, batch=False):
    image = None
    try:
        response = requests.get(image_url_str)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        logging.error(
            'From method getImageFromUrl(image_url_str), {}'.format(e)
        )
        if batch is False:
            abort(408)
        else:
            return None

    if is_flip(image_url_str):
        image = image.transpose(method=Image.FLIP_LEFT_RIGHT)

    return image


def getImageFromPath(image_path, batch=False):
    image = None
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logging.error(
            'From method getImageFromPath(image_path), {}'.format(e)
        )
        if batch is False:
            abort(408)
        else:
            return None

    if is_flip(image_path):
        image = image.transpose(method=Image.FLIP_LEFT_RIGHT)

    return image


@app.route('/api/inference/url', methods=['POST'])
def solution_inference_by_url():
    request_data = request.get_json()
    if 'url' not in request_data.keys():
        abort(405)
    image_url_str = request_data['url']

    image = getImageFromUrl(image_url_str)

    return jsonify(solution_inference(image))


@app.route('/api/inference/path', methods=['POST'])
def solution_inference_by_path():
    request_data = request.get_json()
    if 'path' not in request_data.keys():
        abort(405)
    image_path_str = request_data['path']

    image = getImageFromPath(image_path_str)

    return jsonify(solution_inference(image))

@app.route('/api/solvetasks', methods=['GET'])
def solvetasks():
    timer_start = time.perf_counter()
    serverurl = 'http://192.168.178.22:3001'
    # Login
    s = requests.Session() 
    payload = {"username": "brick_recognition_worker",
    "password": "pass"}
    s.post(serverurl + "/users/login", data=payload)

    # Get Tasks
    data_json = s.get(serverurl + "/tasks/type/5/open").json()
    tasks = data_json["result"]
    
    #Loop Tasks
    task_counter = 0

    for t in tasks:
        task_id = t["id"]
        information_json = json.loads(t["information"])
        if 'image_id' not in information_json.keys():
            # Mark task with error status 
            print(s.put(serverurl + "/tasks/{task_id}/status", data={'id':task_id,'status_id':4}).text)
        image_id = information_json["image_id"]

        if 'imageurl' in information_json.keys():
            url = str(information_json["imageurl"])
            image = getImageFromUrl(url, True)
        elif 'path' in information_json.keys():
            path = str(information_json["path"])
            image = getImageFromPath(path, True)
        else:
            abort(408)

        if image is None:
             print(s.put(serverurl + "/tasks/{task_id}/status", data={'id':task_id,'status_id':4}).text)
             print("task " + str(task_id) + " had no image at the source available - was skipped." )
        else:

            cls_num, conf = brick_classificator(image)

            # Store the result
            s.put(serverurl + "/partimages/{image_id}", data={'id':image_id,'partno':cls_num, 'confidence_partno':float('{:.3f}'.format(conf))})
            # Mark task as completed
            s.put(serverurl + "/tasks/{task_id}/status", data={'id':task_id,'status_id':3})
            print("task " + str(task_id) + " completed." )
            
        task_counter += 1
    timer_end = time.perf_counter()
    return jsonify({"solvedTasks" : task_counter, "elapsedTime": f"{timer_end - timer_start:0.4f} seconds"  })


if __name__ == '__main__':
    args = args_parse()
    app.run(host=args.ip, debug=False, port=args.port)



images = [...]
tensors = [
    torchvision.transforms.ToTensor()(img)
    for img in images
]
input_tensor = torch.stack(tensors, dim=0)
out = model(input_tensor).detach()