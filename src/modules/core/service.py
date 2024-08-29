# built-in dependencies
import traceback
from typing import Optional

# project dependencies
from deepface import DeepFace
from slugify import slugify
import base64
import os

db_path = "database"

# pylint: disable=broad-except


def represent(
    img_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: str,
    img2_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400
    

def add_padding(base64_string):
    return base64_string + '=' * (-len(base64_string) % 4)

def upload(img_path: str, name: str):
    try:
        name_slug = slugify(name)

        print(img_path)

        image = img_path.split(',')
        base64_image = image[1]
        
        # Correct the base64 string padding if necessary
        img_path_padded = add_padding(base64_image)
        
        # Decode the base64 string
        png_recovered = base64.b64decode(img_path_padded)
        
        if not os.path.exists(db_path):
            # Create the directory
            os.makedirs(db_path)
            print(f"Directory '{db_path}' created.")
        else:
            print(f"Directory '{db_path}' already exists.")

        # Save the decoded image as a PNG file
        with open("database/" + name_slug + ".png", "wb") as file:
            file.write(png_recovered)

        DeepFace.find(img_path, db_path="database", model_name="Facenet", detector_backend="mtcnn")
        return {"msg": "success"}, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400
    

def recognize(img_path: str):
    try:

        if not os.path.exists(db_path):
            # Create the directory
            os.makedirs(db_path)
            print(f"Directory '{db_path}' created.")
        else:
            print(f"Directory '{db_path}' already exists.")

        dfs = DeepFace.find(img_path, db_path="database", model_name="Facenet", detector_backend="mtcnn", anti_spoofing=False)

        if len(dfs) == 0:
            # you may consider to return unknown person's image here
            return {"msg": "success", "result" : "unknown"}, 200

        # detected face is coming from parent, safe to access 1st index
        df = dfs[0]

        if df.shape[0] == 0:
            return {"msg": "success", "result" : "unknown"}, 200


        candidate = df.iloc[0]
        target_path = candidate["identity"]
        return {"msg": "success", "result" : target_path}, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400