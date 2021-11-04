import numpy as np
import os
import sys
from timeit import default_timer as time
from argparse import ArgumentParser, Namespace
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from flask import Flask, jsonify, abort, request
from utils.database_connector import DatabaseConnector
import io
import logging
import torch
import torchvision
import json
import time
import requests
import os
import pandas as pd
from PIL import Image, ImageColor
from threading import Lock
import yaml


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


class StandardClassification(object):
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
                torchvision.transforms.Resize(self.input_shape),
                torchvision.transforms.ToTensor()
            ]
        )

    def _preprocess(self, _input_img: Image) -> torch.Tensor:
        inp_tensor = self.transform(_input_img).unsqueeze(0).to(self.device)
        return inp_tensor

    def __call__(self, _img: Image) -> tuple:
        out = self.model(self._preprocess(_img)).detach().to('cpu')[0]
        return self.classes_names[out.argmax()], \
               float('{:.3f}'.format(torch.sigmoid(out).max().numpy()))


app = Flask(__name__)
app_log = logging.getLogger('werkzeug')
app_log.setLevel(logging.ERROR)

brick_classificator = FunctionServingWrapper(
    StandardClassification(
        traced_model_file='calc_models/best_model_partno.pt',
        file_names_file='calc_models/classes_partno.txt',
        device='cpu'
    )
)

resnet_color_classificator_brio = FunctionServingWrapper(
    StandardClassification(
        traced_model_file='calc_models/best_model_color_id.pt',
        file_names_file='calc_models/classes_color_id.txt',
        device='cpu'
    )
)


def read_possible_colors_from_file(file_path):
    global possible_colors
    with open(file_path) as json_file:
        colors = json.load(json_file)
        print(' * Loaded {} possible colors from \'{}\'.'.format(len(colors), file_path))
        return colors


def solution_inference_resnet(img: np.ndarray) -> dict:
    """
    Pipeline inference
    Args:
        img: image in uint8 RGB HWC format
    Returns:
        Dictionary with predicted class name
    """
    global brick_classificator, resnet_color_classificator_brio

    if img is None:
        abort(409)

    partno, partno_conf = brick_classificator(img)
    color_id, color_id_brio = resnet_color_classificator_brio(img)

    return {
        'partno': partno,
        'partno_confidence': float('{:.3f}'.format(partno_conf)),
        'color_id': color_id,
        'color_confidence': float('{:.3f}'.format(color_id_brio))
    }


def concludeBrickProperties(inferences):
    conclusion = {}
    # Get results with highest confidences
    sorted_inferences_partno = sorted(inferences, key=lambda d: d['partno_confidence'], reverse=True)
    sorted_inferences_color_id = sorted(inferences, key=lambda d: d['color_id_confidence'], reverse=True)
    best_color_id = sorted_inferences_color_id[0]['color_id']
    best_color_id_conf = sorted_inferences_color_id[0]['color_id_confidence']

    for inf in sorted_inferences_partno:
        if check_if_color_is_for_partno(inf['partno'], best_color_id):
            conclusion['color_id'] = best_color_id
            conclusion['color_id_confidence'] = best_color_id_conf
            conclusion['partno'] = inf['partno']
            conclusion['partno_confidence'] = inf['partno_confidence']
            print("Predicted partno {}({:2.2%}) Colorid {}({:2.2%})".format(
                inf['partno'], inf['partno_confidence'], best_color_id, best_color_id_conf))
            return conclusion


def check_if_color_is_for_partno(partno, color_id):
    global possible_colors
    for p_color in possible_colors:
        if p_color['no'] == partno:
            break
    else:
        print(" * No possible colors where found for {}".format(partno))
    colors_of_partno = json.loads(p_color['ids'])
    if int(color_id) in colors_of_partno:
        return True
    else:
        return False


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

    return image


def get_database_cursor():
    configfile = 'configuration/lego_brick_recognizer_config.yaml'
    if os.path.exists(configfile):
        with open(configfile, 'r') as conf_f:
            config_dict = yaml.safe_load(conf_f)
        db_connector = DatabaseConnector(config_dict['DATABASE_PROD'])
        return db_connector.get_cursor()


def extract_query_to_object(cur):
    row_headers = [x[0] for x in cur.description]  # this will extract row headers
    rv = cur.fetchall()
    colors = []
    for result in rv:
        colors.append(dict(zip(row_headers, result)))
    return colors


def get_colorinfo_from_db():
    cur = get_database_cursor()
    cur.execute(
        """SELECT color_id, color_name, color_code, color_type, parts_count, year_from, year_to FROM 
        LegoSorterDB.Colors where parts_count > 99 AND color_type IN ('Solid', 'Transparent','Chrome', 'Metallic',
        'Pearl');""")
    return extract_query_to_object(cur)


def get_possible_colors_from_db():
    cur = get_database_cursor()
    cur.execute("""SELECT  p.no as no, JSON_ARRAYAGG(p.color_id) as ids FROM LegoSorterDB.Partdata p 
           GROUP BY no;""")
    return extract_query_to_object(cur)


def write_json_file(object, filename):
    with open(filename, 'w') as outfile:
        json.dump(object, outfile)


@app.route('/api/inference/url', methods=['POST'])
def solution_inference_by_url():
    request_data = request.get_json()
    if 'url' not in request_data.keys():
        abort(405)
    image_url_str = request_data['url']

    image = getImageFromUrl(image_url_str)

    return jsonify(solution_inference_resnet(image))


@app.route('/api/inference/urls', methods=['POST'])
def solution_inference_by_urls():
    timer_start = time.perf_counter()
    image_objects = request.get_json()
    result_inferences = []
    for image_obj in image_objects:
        if 'url' not in image_obj.keys():
            abort(405)

        image = getImageFromUrl(image_obj['url'])
        image_obj['partno'], image_obj['partno_confidence'] = brick_classificator(image)
        image_obj['color_id'], image_obj['color_id_confidence'] = resnet_color_classificator_brio(image)
        if check_if_color_is_for_partno(image_obj['partno'], image_obj['color_id']):
            result_inferences.append(image_obj)
    conclusion = concludeBrickProperties(result_inferences)
    timer_end = time.perf_counter()

    return jsonify({"conclusion": conclusion, "images": result_inferences,
                    "elapsedTime": f"{timer_end - timer_start:0.4f} seconds"})


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

    # Loop Tasks
    task_counter = 0

    for t in tasks:
        task_id = t["id"]
        information_json = json.loads(t["information"])
        if 'image_id' not in information_json.keys():
            # Mark task with error status 
            print(s.put(serverurl + "/tasks/{task_id}/status", data={'id': task_id, 'status_id': 4}).text)
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
            print(s.put(serverurl + "/tasks/{task_id}/status", data={'id': task_id, 'status_id': 4}).text)
            print("task " + str(task_id) + " had no image at the source available - was skipped.")
        else:
            partno, partno_conf = brick_classificator(image)
            color_id, color_id_conf = resnet_color_classificator_brio(image)
            if not check_if_color_is_for_partno(partno, color_id):
                s.put(serverurl + "/tasks/{task_id}/status", data={'id': task_id, 'status_id': 5})
            else:
                # Store the result
                s.put(serverurl + "/partimages/{image_id}",
                      data={'id': image_id, 'partno': partno, 'confidence_partno': float('{:.3f}'.format(partno_conf)),
                            'color_id': color_id, 'color_id_confidence': float('{:.3f}'.format(color_id_conf))})
                # Mark task as completed
                s.put(serverurl + "/tasks/{task_id}/status", data={'id': task_id, 'status_id': 3})

            print("task " + str(task_id) + " completed.")

        task_counter += 1
    timer_end = time.perf_counter()
    return jsonify({"solvedTasks": task_counter, "elapsedTime": f"{timer_end - timer_start:0.4f} seconds"})


@app.route('/api/setup/load_possible_colors', methods=['POST'])
def get_and_save_possible_colors_in_json_file():
    colors = get_possible_colors_from_db()
    write_json_file(colors, 'stored_data/possible_colors.json')

    return jsonify({"task_solved": 'true'})


@app.route('/api/setup/load_colorinfo', methods=['POST'])
def get_and_save_colorinfo_in_json_file():
    colors = get_colorinfo_from_db()
    write_json_file(colors, 'stored_data/colorinfo.json')

    return jsonify({"task_solved": 'true'})


possible_colors = read_possible_colors_from_file('stored_data/possible_colors.json')
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
