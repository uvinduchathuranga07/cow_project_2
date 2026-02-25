from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import pandas as pd
import joblib
import os
import uuid
import random
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

app = Flask(__name__)

BASE_DIR = "models"

YOLO_MODEL_PATH = "best.pt"
FEED_MODEL_PATH = os.path.join(BASE_DIR, "cow_feed_predictor.pkl")

BREED_ENCODER_PATH = os.path.join(BASE_DIR, "breed_encoder.pkl")
ACTIVITY_ENCODER_PATH = os.path.join(BASE_DIR, "activity_encoder.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


_yolo_model = None
_feature_model = None
_feed_model = None
_le_breed = None
_le_activity = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model


def get_feature_model():
    global _feature_model
    if _feature_model is None:
        _feature_model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )
    return _feature_model


def get_feed_model_and_encoders():
    global _feed_model, _le_breed, _le_activity
    if _feed_model is None or _le_breed is None or _le_activity is None:
        _feed_model = joblib.load(FEED_MODEL_PATH)
        _le_breed = joblib.load(BREED_ENCODER_PATH)
        _le_activity = joblib.load(ACTIVITY_ENCODER_PATH)
    return _feed_model, _le_breed, _le_activity


def extract_embedding(face_img):
    feature_model = get_feature_model()

    face_img = cv2.resize(face_img, (224, 224))
    face_img = preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)

    embedding = feature_model.predict(face_img, verbose=0)[0]
    return embedding


def get_cow_embedding_from_image(image_bytes):
    yolo_model = get_yolo_model()

    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = yolo_model(image)

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, boxes[0])
        face_crop = image[y1:y2, x1:x2]

        embedding = extract_embedding(face_crop)
        return embedding

    return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


@app.route("/register", methods=["POST"])
def register():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"].read()
    embedding = get_cow_embedding_from_image(image)

    if embedding is None:
        return jsonify({"error": "No cow detected"}), 400

    return jsonify({
        "embedding": embedding.tolist(),
        "length": len(embedding)
    })


@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    embeddings_list = request.form.get("embeddings")
    if embeddings_list is None:
        return jsonify({"error": "No embeddings provided"}), 400

    # NOTE: eval is unsafe if input is not trusted; better to use json.loads
    embeddings_list = eval(embeddings_list)

    image = request.files["image"].read()
    new_embedding = get_cow_embedding_from_image(image)

    if new_embedding is None:
        return jsonify({"error": "No cow detected"}), 400

    best_match_index = -1
    best_similarity = -1

    for i, emb in enumerate(embeddings_list):
        emb = np.array(emb)
        similarity = cosine_similarity(emb, new_embedding[:len(emb)])

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_index = i

    return jsonify({
        "match_index": int(best_match_index),
        "similarity": float(best_similarity),
        "matched": True if best_similarity > 0.70 else False
    })


@app.route("/predict", methods=["POST"])
def predict_from_image():
    img_path = None

    try:
        if "image" not in request.files:
            return jsonify({"error": "Image required"}), 400

        file = request.files["image"]
        filename = str(uuid.uuid4()) + ".jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        feed_model, le_breed, le_activity = get_feed_model_and_encoders()
        yolo_model = get_yolo_model()

        cow_breed = request.form.get("breed").strip().title()
        cow_age = float(request.form.get("age"))
        milk_yield = float(request.form.get("milk_yield"))
        activity = request.form.get("activity").strip().title()

        if cow_breed not in le_breed.classes_:
            return jsonify({
                "error": f"Invalid breed. Allowed: {list(le_breed.classes_)}"
            }), 400

        if activity not in le_activity.classes_:
            return jsonify({
                "error": f"Invalid activity. Allowed: {list(le_activity.classes_)}"
            }), 400

        encoded_breed = le_breed.transform([cow_breed])[0]
        encoded_activity = le_activity.transform([activity])[0]

        results = yolo_model.predict(
            source=img_path,
            conf=0.05,
            save=False,
            verbose=False
        )

        cow_weight = None

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            cls_id = int(r.boxes.cls[0])
            class_name = yolo_model.names[cls_id]

            try:
                if "-kg" in class_name:
                    weight_part = class_name.replace("-kg", "")
                    min_w, max_w = weight_part.split("-")
                    min_w = float(min_w) * 3
                    max_w = float(max_w) * 3
                    cow_weight = round(random.uniform(min_w, max_w), 2)
            except Exception:
                pass

            break

        if cow_weight is None:
            return jsonify({"error": "Cow not detected"}), 400

        feed_input = pd.DataFrame([{
            "Cow Breed": encoded_breed,
            "Cow Age (months)": cow_age,
            "Cow Weight (kg)": cow_weight,
            "Milk Yield (L/day)": milk_yield,
            "Activity Level": encoded_activity
        }])

        daily_feed = float(feed_model.predict(feed_input)[0])

        os.remove(img_path)

        return jsonify({
            "mode": "image",
            "cow_weight_kg": cow_weight,
            "daily_feed_kg": round(daily_feed, 2)
        })

    except Exception as e:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": str(e)}), 500


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        feed_model, le_breed, le_activity = get_feed_model_and_encoders()

        data = request.get_json()

        cow_breed = data["breed"].strip().title()
        cow_age = float(data["age"])
        cow_weight = float(data["weight"])
        milk_yield = float(data["milk_yield"])
        activity = data["activity"].strip().title()

        if cow_breed not in le_breed.classes_:
            return jsonify({
                "error": f"Invalid breed. Allowed: {list(le_breed.classes_)}"
            }), 400

        if activity not in le_activity.classes_:
            return jsonify({
                "error": f"Invalid activity. Allowed: {list(le_activity.classes_)}"
            }), 400

        encoded_breed = le_breed.transform([cow_breed])[0]
        encoded_activity = le_activity.transform([activity])[0]

        feed_input = pd.DataFrame([{
            "Cow Breed": encoded_breed,
            "Cow Age (months)": cow_age,
            "Cow Weight (kg)": cow_weight,
            "Milk Yield (L/day)": milk_yield,
            "Activity Level": encoded_activity
        }])

        daily_feed = float(feed_model.predict(feed_input)[0])

        return jsonify({
            "mode": "manual",
            "cow_weight_kg": cow_weight,
            "daily_feed_kg": round(daily_feed, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    # Try to report whether models have been loaded (not force-load them)
    return jsonify({
        "status": "ok",
        "yolo_loaded": _yolo_model is not None,
        "resnet_loaded": _feature_model is not None,
        "feed_model_loaded": _feed_model is not None
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # In production with gunicorn/uwsgi, disable debug and let the WSGI server run the app
    app.run(host="0.0.0.0", port=port)