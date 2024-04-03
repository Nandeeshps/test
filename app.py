from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

@app.route('/', methods=['GET'])
def home():
    # Render a home page with an upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Convert the image to the correct size and format for your model
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))  # Example size, adjust to your model's input
        image = np.expand_dims(image, axis=0)
        
        # Make a prediction
        predictions = model.predict(image)
        # Assuming your model returns a class index, you might need to adjust this
        predicted_class = np.argmax(predictions, axis=1)
        
        # Map the predicted class index to the actual class name
        class_names = ['Class1', 'Class2', 'Class3']  # Example class names, adjust to your model's classes
        result = class_names[predicted_class[0]]
        
        return f'Predicted Skin Cancer Type: {result}'

if __name__ == '__main__':
    app.run(debug=True)
