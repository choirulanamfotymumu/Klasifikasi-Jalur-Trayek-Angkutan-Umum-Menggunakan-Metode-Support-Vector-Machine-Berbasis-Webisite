# angkot_classifier_app/app.py

import os
from flask import Flask, render_template, request, jsonify
import joblib
import cv2
import numpy as np
from PIL import Image
import io
from werkzeug.utils import secure_filename

# --- Configuration ---
# Define the path to the saved SVM model
MODEL_PATH = 'svm_angkot_classifier_model.joblib'

# Define approximate coordinates for key points in Sorong
# These are used to construct Google Maps URLs.
# Format: (latitude, longitude)
SORONG_COORDS = {
    # --- Precise user-provided coordinates for Trayek A, B, E, H ---
    "Terminal Remu": (-0.8852760295507047, 131.28726767029673),
    "HBM": (-0.8804563378855831, 131.28698231627826),
    "Jl.Mesjid Raya": (-0.879684, 131.282728),
    "Yohan": (-0.882720, 131.274619),
    "Saga": (-0.876447, 131.257917),
    "Halte Dom": (-0.876144, 131.243884),
    "Kampung Baru": (-0.872322, 131.249905),
    "Rufei": (-0.863577, 131.249314),
    "Pelabuhan": (-0.877237, 131.247448),
    "Pasar Boswesen": (-0.862701, 131.247016),
    "Surya": (-0.865334, 131.252349),
    "Pasar Remu": (-0.887187, 131.285956),
    "Terminal Samping Maranata": (-0.884523, 131.287556),
    "Kelapa 2": (-0.879020, 131.291590),
    "Kios Anda": (-0.875458, 131.296842),
    "Malanu lokasi": (-0.877465, 131.300205),
    "KPR Pepabri": (-0.879419, 131.301790),
    "Malanu kampung": (-0.879916, 131.303173),
    "Arteri": (-0.878917, 131.314779),
    "Smp 4": (-0.873839, 131.311987),
    "KM.7": (-0.885910, 131.288622),
    "KM.8": (-0.890038, 131.296161),
    "KM.9": (-0.888236, 131.308177),
    "KM.10": (-0.891233, 131.315601),
    "Depan Batalion": (-0.896025, 131.319080),
    "KM.12": (-0.902653, 131.323563),
    "KM.12 Masuk": (-0.899500, 131.328436), # Using this for the "KM.12 Masuk" segment
    "Gunung Jufri": (-0.885354, 131.322995),

    # --- General reference points (not used in current routes, but kept for context) ---
    "Terminal Penumpang": (-0.887928, 131.267970), # General reference, might be near Terminal Remu
    "Ramayana": (-0.88204, 131.27553), # Mal Sorong (already used in Trayek A)
    "Jl. Jend. Sudirman": (-0.875, 131.258),
    "Jl. Rumberpon": (-0.880, 131.259),
    "Jl. Basuki Rahmat": (-0.885, 131.260),
    "Jl. A. Yani": (-0.870, 131.250),
    "Jl. Yos Sudarso": (-0.865, 131.245),
    "Jl. Arfak": (-0.860, 131.248),
    "Jl. Sam Ratulangi": (-0.855, 131.252),
    "Jl. Diponegoro": (-0.850, 131.255),
    "Pasar Sentral": (-0.878, 131.257),
    "Jl. Bubara": (-0.870, 131.255),
    "Jl. Pahlawan": (-0.880, 131.250),
    "Jl. Pramuka": (-0.882, 131.255),
    "Jl. Rajawali": (-0.890, 131.262),
    "Jl. F. Kaisepo": (-0.895, 131.265),
    "Jl. Kurani": (-0.900, 131.268),
    "Jl. F. Kalasuat": (-0.905, 131.270),
    "Malanu": (-0.910, 131.275), # General Malanu, different from "Malanu kampung"
    "Jl. Mamberamo": (-0.915, 131.280),
    "Pemukiman Misi": (-0.920, 131.285),
    "Jl. Pendidikan": (-0.908, 131.272),
    "Jl. Kasturi": (-0.888, 131.258),
    "Hutan Lindung": (-0.910, 131.290),
    "Moyo": (-0.910, 131.288),
}

# Define angkot routes using the approximate coordinates
# These routes are lists of (latitude, longitude) tuples.
ANGKOT_ROUTES = {
    "A": {
        "name": "Trayek A",
        "route": [
            SORONG_COORDS["Terminal Remu"],
            SORONG_COORDS["HBM"],
            SORONG_COORDS["Jl.Mesjid Raya"],
            SORONG_COORDS["Yohan"],
            SORONG_COORDS["Saga"],
            SORONG_COORDS["Halte Dom"],
            SORONG_COORDS["Kampung Baru"],
            SORONG_COORDS["Rufei"],
            SORONG_COORDS["Terminal Remu"] # Putar balik ke terminal pasar (menggunakan Terminal Remu)
        ]
    },
    "B": {
        "name": "Trayek B",
        "route": [
            SORONG_COORDS["Terminal Remu"],
            SORONG_COORDS["Saga"],
            SORONG_COORDS["Pelabuhan"],
            SORONG_COORDS["Halte Dom"],
            SORONG_COORDS["Pasar Boswesen"],
            SORONG_COORDS["Rufei"],
            SORONG_COORDS["Surya"],
            SORONG_COORDS["Kampung Baru"],
            SORONG_COORDS["Yohan"],
            SORONG_COORDS["Pasar Remu"]
        ]
    },
    "E": {
        "name": "Trayek E",
        "route": [
            SORONG_COORDS["Terminal Samping Maranata"],
            SORONG_COORDS["Kelapa 2"],
            SORONG_COORDS["Kios Anda"],
            SORONG_COORDS["Malanu lokasi"],
            SORONG_COORDS["KPR Pepabri"],
            SORONG_COORDS["Malanu kampung"],
            SORONG_COORDS["Arteri"],
            SORONG_COORDS["Smp 4"],
            SORONG_COORDS["Terminal Samping Maranata"] # Putar balik ke terminal
        ]
    },
    "H": {
        "name": "Trayek H",
        "route": [
            SORONG_COORDS["Terminal Remu"], # Menggunakan Terminal Remu sebagai titik awal
            SORONG_COORDS["KM.7"],
            SORONG_COORDS["KM.8"],
            SORONG_COORDS["KM.9"],
            SORONG_COORDS["KM.10"],
            SORONG_COORDS["Depan Batalion"],
            SORONG_COORDS["KM.12"], # Mengganti Moyo dengan KM.12
            SORONG_COORDS["KM.12 Masuk"], # Titik masuk KM.12
            SORONG_COORDS["Gunung Jufri"], # Mengganti KM.13 dengan Gunung Jufri
            SORONG_COORDS["KM.10"], # Balik lagi ke kilo 10
            SORONG_COORDS["Terminal Remu"] # Balik lagi ke terminal pasar
        ]
    }
}


# --- Load the SVM model (loaded once when the app starts) ---
loaded_svm_model = None
try:
    loaded_svm_model = joblib.load(MODEL_PATH)
    print(f"SVM model loaded successfully from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    print("Please ensure the model file exists and the path is correct.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")


# --- Function to generate Google Maps URL for a given route ---
def generate_google_maps_url(route_key):
    """
    Generates a Google Maps URL for directions based on the defined angkot routes.
    The URL will include origin, destination, and waypoints.
    """
    route_data = ANGKOT_ROUTES.get(route_key)
    if not route_data:
        return None

    coordinates = route_data["route"]
    if not coordinates:
        return None

    # Google Maps URL format: https://www.google.com/maps/dir/?api=1&origin=LAT,LON&destination=LAT,LON&waypoints=LAT,LON|LAT,LON...&travelmode=driving
    
    origin_lat, origin_lon = coordinates[0]
    destination_lat, destination_lon = coordinates[-1]

    # Format waypoints (all points between origin and destination)
    waypoints = []
    if len(coordinates) > 2:
        for lat, lon in coordinates[1:-1]:
            waypoints.append(f"{lat},{lon}")
    waypoints_str = "|".join(waypoints)

    base_url = "https://www.google.com/maps/dir/?api=1"
    url_params = f"origin={origin_lat},{origin_lon}&destination={destination_lat},{destination_lon}"
    
    if waypoints_str:
        url_params += f"&waypoints={waypoints_str}"
    
    url_params += "&travelmode=driving" # Assuming angkot uses driving mode

    return f"{base_url}&{url_params}"


# --- Initialize Flask Application ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/' # For temporary image storage
app.secret_key = 'your_super_secret_key_here' # **CHANGE THIS IN PRODUCTION**

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main HTML page with the file upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, prediction, and returns results."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # If user does not select a file, browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # If file is valid and model is loaded, proceed with processing
        if file and allowed_file(file.filename) and loaded_svm_model:
            try:
                # Read the image file
                image_bytes = file.read()
                img = Image.open(io.BytesIO(image_bytes))

                # Convert PIL Image to OpenCV format (NumPy array)
                # Ensure color consistency (e.g., RGB to BGR if OpenCV expects BGR)
                if img.mode == 'RGBA':
                    img = img.convert('RGB') # Convert RGBA to RGB if necessary
                img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Preprocess the image (resize, grayscale, flatten) - must match training preprocessing
                img_resized = cv2.resize(img_cv2, (128, 128))
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_flattened = img_gray.flatten()

                # Reshape for prediction (add batch dimension)
                img_for_prediction = img_flattened.reshape(1, -1)

                # --- Make prediction using the loaded model ---
                prediction = loaded_svm_model.predict(img_for_prediction)
                predicted_route_key = str(prediction[0]) # Get the prediction result as a string

                print(f"Image processed. Predicted route: {predicted_route_key}")

                # --- Generate Google Maps URL for the predicted route ---
                google_maps_url = generate_google_maps_url(predicted_route_key)
                
                if google_maps_url:
                    print(f"Google Maps URL generated for route '{predicted_route_key}': {google_maps_url}")
                else:
                    print(f"Warning: Could not generate Google Maps URL for route '{predicted_route_key}'.")


                # Return the prediction result and Google Maps URL
                return jsonify({
                    'prediction': predicted_route_key,
                    'google_maps_url': google_maps_url
                })


            except Exception as e:
                print(f"An error occurred during image processing or prediction: {e}")
                return jsonify({'error': f'Error processing image or making prediction: {e}'}), 500

        elif loaded_svm_model is None:
             return jsonify({'error': 'Model tidak dimuat. Tidak dapat melakukan prediksi.'}), 500
        else:
            # Handle invalid file type
            return jsonify({'error': 'Jenis file tidak diizinkan. Harap unggah gambar (png, jpg, jpeg, gif).'}), 400

    # If not a POST request, redirect or return error
    return jsonify({'error': 'Metode tidak diizinkan'}), 405


# --- Add the standard Flask development server run block ---
if __name__ == '__main__':
    # Hapus file lama di folder uploads saat startup (opsional, untuk kebersihan)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Ensure folder exists
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path) # Remove the file
        except Exception as e:
            print(f"Error removing old file {file_path}: {e}")

    # Running with debug=True is helpful during development for automatic reloading and detailed error messages.
    # host='0.0.0.0' makes the server accessible externally, which is necessary in environments like Google Colab.
    # port=5000 is a common port for Flask development servers.
    # threaded=True allows the development server to handle multiple requests concurrently, which is useful for AJAX requests.
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
