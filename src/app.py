# 3rd parth dependencies
from flask import Flask
from flask_cors import CORS

# project dependencies
from deepface import DeepFace
from modules.core.routes import blueprint
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app
