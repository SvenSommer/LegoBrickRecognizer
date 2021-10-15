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
               float(torch.sigmoid(out).max().numpy())


class BrickColorEstimation(object):
    base_by_materials = dict()

    def __init__(self,
                 traced_model_file: str,
                 colors_types_names_file: str,
                 device: str = 'cpu',
                 possible_colors_json_file: str = 'possible_colors.json',
                 colorinfo_json_file: str = 'colorinfo.json'):
        self.color_type_model = torch.jit.load(traced_model_file)
        self.color_type_model.eval()
        self.color_type_model = self.color_type_model.to(device)
        self.input_shape = (224, 224)
        self.device = device

        self._read_colorinfo_from_file(colorinfo_json_file)
        self._read_possible_colors_from_file(possible_colors_json_file)
        self.base_by_materials = self._prepare_table(self.colors)

        with open(colors_types_names_file, 'r') as f:
            self.color_types_names = [line.rstrip() for line in f]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.input_shape),
                torchvision.transforms.ToTensor()
            ]
        )

    def _read_possible_colors_from_file(self, file_path):

        with open(file_path) as json_file:
            self.possible_colors = json.load(json_file)
            print('[INFO] Loaded {} possible colors from \'{}\'.'.format(len(self.possible_colors), file_path))

    def _read_colorinfo_from_file(self, file_path):
        with open(file_path) as json_file:
            self.colors = json.load(json_file)
            print('[INFO] Loaded information of {} colors  from \'{}\'.'.format(len(self.colors), file_path))

    @staticmethod
    def _prepare_table(colors, colors_of_partno=None):
        materials = set()
        for dic in colors:
            materials.add(dic['color_type'])

        result_array = {
            material_name: {'color_id': [], 'rgb': []}
            for material_name in materials
        }

        for i in range(0, len(colors)):
            if colors_of_partno is None or (colors[i]['color_id'] in colors_of_partno):
                hex_cl = '#' + str(colors[i]['color_code']).replace(' ', '')
                if hex_cl == '#0':
                    hex_cl = '#000000'
                color_id = colors[i]['color_id']

                rgb = np.array(ImageColor.getrgb(hex_cl))
                if rgb.shape != (3,):
                    continue

                material = colors[i]['color_type']
                result_array[material]['color_id'].append(color_id)
                result_array[material]['rgb'].append(rgb)

        if len(result_array) > 0:
            return result_array
        else:
            print("Problem with possible_colors {}".format(possible_colors))
            return self._prepare_table(colors)

    def _preprocess(self, _input_img: Image) -> torch.Tensor:
        inp_tensor = self.transform(_input_img).unsqueeze(0).to(self.device)
        return inp_tensor

    @staticmethod
    def find_nearest_color_by_cielab(_base_colors, _input_vec):
        dists = [
            delta_e_cie2000(
                convert_color(sRGBColor(*_input_vec), LabColor),
                convert_color(sRGBColor(*cb), LabColor),
            )
            for cb in _base_colors
        ]
        return np.array(dists).argmin(), np.amin(dists)

    def get_reduced_base_by_partno(self, partno):
        for x in self.possible_colors:
            if x['no'] == partno:
                break
        else:
            print("No possible colors where found for {}".format(partno))
            return self.base_by_materials

        # print("Found colors for partno '{}': {}".format(partno, x['ids']))
        return self._prepare_table(self.colors, json.loads(x['ids']))

    def get_reduced_color_types(self, table):
        reduced_colors = []
        for m in table:
            if table[m]['color_id']:
                reduced_colors.append(m)
        return reduced_colors

    def __call__(self, _img: Image, partno: str) -> tuple:
        reduced_base_by_materials = self.get_reduced_base_by_partno(partno)
        # TODO: Fix bug if no possible colors for a predicted color_type are available
        # e.g. http://192.168.178.53:5000/extracted_images/34/run_id_34_id_12972_position_BOTTOM_side_BRIO_MIDDLE.jpg
        reduced_color_types = self.get_reduced_color_types(reduced_base_by_materials)
        color_types_prediction = self.color_type_model(
            self._preprocess(_img)).detach().to('cpu')[0]
        # print("color_types_prediction: {}".format(color_types_prediction))
        # print("self.color_types_names: {}".format(self.color_types_names))
        # print("reduced_color_types: {}".format(reduced_color_types))
        color_type_idx = color_types_prediction.argmax()

        type_name = self.color_types_names[color_type_idx]

        # Dirty Workaround:
        if len(reduced_base_by_materials[type_name]['rgb']) == 0:
            print("color_type was {} now changing to Solid".format(type_name))
            type_name = "Solid"
        if len(reduced_base_by_materials[type_name]['rgb']) == 0:
            print("color_type was {} now changing to Transparent".format(type_name))
            type_name = "Transparent"
        if len(reduced_base_by_materials[type_name]['rgb']) == 0:
            print("color_type was {} now changing to Chrome".format(type_name))
            type_name = "Chrome"
        if len(reduced_base_by_materials[type_name]['rgb']) == 0:
            print("color_type was {} now changing to Metallic".format(type_name))
            type_name = "Metallic"
        if len(reduced_base_by_materials[type_name]['rgb']) == 0:
            print("color_type was {} now changing to Pearl".format(type_name))
            type_name = "Pearl"

        vec_img = np.array(_img.convert('RGB'))
        xc, yc = (vec_img.shape[1] // 2, vec_img.shape[0] // 2)
        r = 5

        crop = vec_img[yc - r:yc + r, xc - r:xc + r].copy()
        avg_rgb = crop.mean(axis=(0, 1))


        # print("type_name: {}".format(type_name))
        # print("reduced_base_by_materials: {}".format(reduced_base_by_materials))
        sub_idx, dist = self.find_nearest_color_by_cielab(
            reduced_base_by_materials[type_name]['rgb'],
            avg_rgb
        )

        predicted_color_id = \
            int(reduced_base_by_materials[type_name]['color_id'][sub_idx])
        return predicted_color_id, type_name, dist


app = Flask(__name__)
app_log = logging.getLogger('werkzeug')
app_log.setLevel(logging.ERROR)

brick_classificator = FunctionServingWrapper(
    BrickClassification(
        traced_model_file='traced_best_model.pt',
        file_names_file='classes.txt',
        device='cpu'
    )
)
brick_color_estimator = FunctionServingWrapper(
    BrickColorEstimation(
        traced_model_file='brick_color_type.torchscript.pt',
        colors_types_names_file='color_types.txt',
        device='cpu',
        possible_colors_json_file='possible_colors.json'
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

    partno, partno_conf = brick_classificator(img)
    color_id, color_type, dist = brick_color_estimator(img, partno)

    return {
        'partno': partno,
        'partno_confidence': float('{:.3f}'.format(partno_conf)),
        'color_id': color_id,
        'color_type': color_type,
        'color_distance': float('{:.2f}'.format(dist))
    }


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

    return jsonify(solution_inference(image))


@app.route('/api/setup/load_possible_colors', methods=['POST'])
def get_and_save_possible_colors_in_json_file():
    colors = get_possible_colors_from_db()
    write_json_file(colors, 'possible_colors.json')

    return jsonify({"task_solved": 'true'})


@app.route('/api/setup/load_colorinfo', methods=['POST'])
def get_and_save_colorinfo_in_json_file():
    colors = get_colorinfo_from_db()
    write_json_file(colors, 'colorinfo.json')

    return jsonify({"task_solved": 'true'})


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
            color_id, color_type, dist  = brick_color_estimator(image, partno)

            # Store the result
            s.put(serverurl + "/partimages/{image_id}",
                  data={'id': image_id, 'partno': partno, 'confidence_partno': float('{:.3f}'.format(partno_conf)),
                        'color_id': color_id, 'color_distance': float('{:.2f}'.format(dist))})
            # Mark task as completed
            s.put(serverurl + "/tasks/{task_id}/status", data={'id': task_id, 'status_id': 3})
            print("task " + str(task_id) + " completed.")

        task_counter += 1
    timer_end = time.perf_counter()
    return jsonify({"solvedTasks": task_counter, "elapsedTime": f"{timer_end - timer_start:0.4f} seconds"})


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
