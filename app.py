import os
import classify_video
import Videos_main
import keras
from flask import Flask, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import qrcode

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
print("hello")
# Load your pre-trained machine learning model
model = keras.models.load_model("cnn_lstm_model_PRO.hdf5")
print("model is loaded")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def classify_video_func(video_path):
    # Placeholder for actual classification
    if Videos_main.main(video_path, model) == "Violence":
        return True
    return False


def create_qr_code(data, filepath):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save(filepath)


@app.route('/')
def index():
    # Use the server's IP address instead of localhost
    server_ip = request.host.split(':')[0]
    server_port = 5000  # Default Flask port
    url = f'http://{server_ip}:{server_port}/'

    qr_code_path = os.path.join('static', 'qr_code.png')
    # create_qr_code(url, qr_code_path)
    return render_template('index.html', qr_code_path=qr_code_path)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        is_violent = classify_video_func(filepath)

        return render_template('result.html', is_violent=is_violent)
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Run the app on the local network
    app.run(host='0.0.0.0', port=5000, debug=True)

