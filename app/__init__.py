import os
from flask import Flask
from .blueprints.main import bp as main_bp

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-not-secure"),
        MODEL_PATH=os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "spam_pipeline.joblib")),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        UPLOAD_FOLDER=os.path.join(os.path.dirname(__file__), "..", "uploads"),
    )
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    app.register_blueprint(main_bp)
    return app
