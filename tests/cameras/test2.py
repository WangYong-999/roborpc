import cv2
import numpy as np
from multiprocessing import Pool
from threading import Thread
import time
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
from io import BytesIO
import base64
import os

app = Flask(__name__)
socketio = SocketIO(app)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def apply_filter(self, filter_type):
        if filter_type == 'grayscale':
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif filter_type == 'blur':
            return cv2.GaussianBlur(self.image, (15, 15), 0)
        elif filter_type == 'edge':
            return cv2.Canny(self.image, 100, 200)
        elif filter_type == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(self.image, -1, kernel)
        elif filter_type == 'sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            return cv2.transform(self.image, sepia_filter)
        else:
            return self.image


def apply_filter_multithread(processor, filter_type):
    filtered_image = processor.apply_filter(filter_type)
    return filtered_image


def apply_filter_multiprocess(image, filter_type):
    processor = ImageProcessor(image)
    filtered_image = processor.apply_filter(filter_type)
    return filtered_image


def process_image(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output_images = []

    filter_types = ['grayscale', 'blur', 'edge', 'sharpen', 'sepia']

    # Multithreading
    start_time = time.time()
    threads = []
    for filter_type in filter_types:
        t = Thread(target=apply_filter_multithread, args=(ImageProcessor(image), filter_type))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    multithreading_time = time.time() - start_time

    # Multiprocessing
    start_time = time.time()
    pool = Pool(processes=len(filter_types))
    results = [pool.apply_async(apply_filter_multiprocess, args=(image, filter_type)) for filter_type in filter_types]
    output_images = [res.get() for res in results]
    pool.close()
    pool.join()
    multiprocessing_time = time.time() - start_time

    # Convert images to base64 strings and save them
    encoded_images = []
    saved_image_paths = []
    for i, img in enumerate(output_images):
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        encoded_images.append(encoded_image)

        # Save the image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{filter_types[i]}.jpg')
        cv2.imwrite(image_path, img)
        saved_image_paths.append(image_path)

    return encoded_images, multithreading_time, multiprocessing_time, saved_image_paths


def real_time_processing(image_data, filter_type):
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processor = ImageProcessor(image)

    if filter_type == 'grayscale':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gray = cv2.cvtColor(np.array([[image[i, j]]], dtype=np.uint8), cv2.COLOR_BGR2GRAY)
                image[i, j] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # Emit the current state of the image
                _, buffer = cv2.imencode('.jpg', image)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('image_update', {'image': encoded_image})
                time.sleep(0.001)  # Simulate processing time
    elif filter_type == 'blur':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                blurred = cv2.GaussianBlur(image[:i + 1, :j + 1], (15, 15), 0)
                image[:i + 1, :j + 1] = blurred[:i + 1, :j + 1]
                # Emit the current state of the image
                _, buffer = cv2.imencode('.jpg', image)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('image_update', {'image': encoded_image})
                time.sleep(0.001)  # Simulate processing time
    # Add similar processing for other filters as needed


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)


    return render_template('result.html', output_images=output_images, multithreading_time=multithreading_time,
                           multiprocessing_time=multiprocessing_time, saved_image_paths=saved_image_paths)


@socketio.on('start_processing')
def handle_start_processing(data):
    image_data = base64.b64decode(data['image_data'])
    filter_type = data['filter_type']
    Thread(target=real_time_processing, args=(image_data, filter_type)).start()


if __name__ == '__main__':
    image_data = np.fromfile('test2.jpg', np.uint8)
    output_images, multithreading_time, multiprocessing_time, saved_image_paths = process_image(image_data)
    print(multithreading_time, multiprocessing_time)
