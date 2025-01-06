from flask import Flask, request, jsonify, send_file, render_template, url_for
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/static', static_folder='.')
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect-edges', methods=['POST'])
def detect_edges():
    # Check if the file is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read the image using OpenCV
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the result to 0-255
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

    # Save the result image
    result_path = os.path.join(RESULT_FOLDER, f'sobel_{file.filename}')
    cv2.imwrite(result_path, sobel_combined)

    # Return the result image
    return send_file(result_path, mimetype='image/png')

@app.route('/analyze-histogram', methods=['POST'])
def analyze_histogram():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Simpan file yang diunggah
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Baca gambar menggunakan OpenCV
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Ubah gambar ke grayscale jika perlu
    if len(image.shape) == 3:  # Cek jika gambar memiliki 3 channel (RGB)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image

    # Simpan gambar grayscale
    grayscale_path = os.path.join(RESULT_FOLDER, f'grayscale_{file.filename}')
    cv2.imwrite(grayscale_path, grayscale_image)

    # Hitung histogram
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    histogram = histogram / histogram.sum()

    # Buat grafik histogram
    plt.figure()
    plt.title("Normalized Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(histogram, color='blue')
    plt.xlim([0, 256])

    # Simpan grafik histogram
    histogram_path = os.path.join(RESULT_FOLDER, f'histogram_{file.filename}.png')
    plt.savefig(histogram_path)
    plt.close()

    # Tampilkan gambar grayscale dan histogram di halaman web
    return render_template(
        'index.html',
        histogram_url=url_for('static', filename=f'results/histogram_{file.filename}.png'),
        grayscale_url=url_for('static', filename=f'results/grayscale_{file.filename}'),
        image_url=url_for('static', filename=f'uploads/{file.filename}')
    )



@app.route('/segment-image', methods=['POST'])
def segment_image():
    # Check if the file is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read the image using OpenCV
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the segmented image
    segmented_path = os.path.join(RESULT_FOLDER, f'segmented_{file.filename}')
    cv2.imwrite(segmented_path, segmented)

    # Return the segmented image
    return send_file(segmented_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
