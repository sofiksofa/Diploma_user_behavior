from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'saved_models'
app.config['ALLOWED_EXTENSIONS'] = {'pth', 'pt'}

# Загрузка базовой архитектуры модели
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNN()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_weights():
    if 'weights' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['weights']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        try:
            model.load_state_dict(torch.load(save_path))
            return jsonify({'message': 'Weights uploaded successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = torch.randn(1, 10)  # Пример входных данных
        with torch.no_grad():
            output = model(data)
        return render_template('result.html', 
                             prediction=output.numpy().tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
