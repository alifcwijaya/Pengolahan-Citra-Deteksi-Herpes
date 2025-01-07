from flask import Flask, request, jsonify, send_file, render_template, url_for
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 

app = Flask(__name__, static_url_path='/static', static_folder='.')
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect-edges', methods=['POST'])
def detect_edges():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only jpg, jpeg, and png are allowed.'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    edge_threshold = int(request.form.get('edge_threshold', 127))

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    _, sobel_thresholded = cv2.threshold(sobel_combined, edge_threshold, 255, cv2.THRESH_BINARY)

    result_path = os.path.join(RESULT_FOLDER, f'sobel_{file.filename}')
    cv2.imwrite(result_path, sobel_thresholded)

    return send_file(result_path, mimetype='image/png')

@app.route('/analyze-histogram', methods=['POST'])
def analyze_histogram():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only jpg, jpeg, and png are allowed.'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image

    grayscale_path = os.path.join(RESULT_FOLDER, f'grayscale_{file.filename}')
    cv2.imwrite(grayscale_path, grayscale_image)

    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    histogram = histogram / histogram.sum()

    plt.figure()
    plt.title("Normalized Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(histogram, color='blue')
    plt.xlim([0, 256])

    histogram_path = os.path.join(RESULT_FOLDER, f'histogram_{file.filename}.png')
    plt.savefig(histogram_path)
    plt.close()

    return render_template(
        'index.html',
        histogram_url=url_for('static', filename=f'results/histogram_{file.filename}.png'),
        grayscale_url=url_for('static', filename=f'results/grayscale_{file.filename}'),
        image_url=url_for('static', filename=f'uploads/{file.filename}')
    )

@app.route('/segment-image', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only jpg, jpeg, and png are allowed.'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segment_threshold = int(request.form.get('segment_threshold', 127))

    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    segmented_path = os.path.join(RESULT_FOLDER, f'segmented_{file.filename}')
    cv2.imwrite(segmented_path, segmented)

    return send_file(segmented_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)