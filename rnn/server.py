from pathlib import Path
import sys

from bottle import Bottle, request, response, run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rnn.predict import (
    load_categories,
    load_config,
    load_model,
    predict as run_prediction,
)

app = Bottle()
_config = load_config()
_categories = load_categories(_config["data_dir"], _config["data_files"])
_checkpoint = _config["dump_dir_with_host"] / _config["output_file"]
_fallback = _config["dump_dir"] / _config["output_file"]
_model = load_model(len(_categories), _checkpoint, _fallback)
_DEFAULT_TOP_N = 3


def _as_json(payload, status=200):
    response.status = status
    response.content_type = "application/json"
    return payload


def _build_payload(name: str, n_predictions: int):
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("Name must be a non-empty string.")
    predictions = run_prediction(_model, _categories, cleaned_name, n_predictions)
    top_category = predictions[0]["category"] if predictions else None
    return {
        "name": cleaned_name,
        "predicted_category": top_category,
        "predictions": predictions,
    }


@app.get("/predict")
def predict_handler():
    name = request.query.get("name", "")
    raw_top = request.query.get("n", _DEFAULT_TOP_N)
    try:
        top_n = max(1, int(raw_top))
    except (TypeError, ValueError):
        return _as_json({"error": "Parameter 'n' must be an integer."}, status=400)
    try:
        payload = _build_payload(name, top_n)
    except ValueError as exc:
        return _as_json({"error": str(exc)}, status=400)
    return _as_json(payload)


@app.get("/<name>")
def legacy_handler(name):
    if name == "favicon.ico":
        return _as_json({"error": "Not Found"}, status=404)
    try:
        payload = _build_payload(name, _DEFAULT_TOP_N)
    except ValueError as exc:
        return _as_json({"error": str(exc)}, status=400)
    return _as_json(payload)


if __name__ == "__main__":
    run(app=app, host="localhost", port=5533)
