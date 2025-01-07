from flask import Flask, request, jsonify, send_file, render_template, url_for
import cv2
import numpy as np
import os
import io

app = Flask(__name__, static_url_path='/static', static_folder='static')  # Gunakan folder static
UPLOAD_FOLDER = '/tmp/uploads'  # Gunakan folder sementara di serverless environment
RESULT_FOLDER = '/tmp/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

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

    # Membaca file gambar
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Deteksi tepi menggunakan Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))  # Normalisasi

    # Encode hasil sebagai gambar PNG
    _, buffer = cv2.imencode('.png', sobel_combined)
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='sobel_edges.png'
    )

@app.route('/analyze-histogram', methods=['POST'])
def analyze_histogram():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Membaca file gambar
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Konversi ke grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hitung histogram
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    histogram = histogram / histogram.sum()  # Normalisasi

    # Buat file histogram menggunakan Matplotlib
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("Normalized Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(histogram, color='blue')
    plt.xlim([0, 256])

    # Simpan hasil grafik ke buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(
        buf,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='histogram.png'
    )

@app.route('/segment-image', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Membaca file gambar
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Segmentasi menggunakan thresholding Otsu
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encode hasil sebagai gambar PNG
    _, buffer = cv2.imencode('.png', segmented)
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='segmented_image.png'
    )

if __name__ == '__main__':
    app.run()
