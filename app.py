from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('skin_cancer_detection_model.h5') # Load your trained model

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/precaution', methods=['GET'])
def precaution():
    return render_template('precaution.html')

@app.route('/information', methods=['GET', 'POST'])
def information():
    if request.method == 'POST':
        selected_location = request.form['location']

        # Redirect to the page corresponding to the selected location
        if selected_location == 'bangalore':
            return redirect(url_for('bangalore_page'))
        elif selected_location == 'chennai':
            return redirect(url_for('chennai_page'))
        elif selected_location == 'hyderabad':
            return redirect(url_for('hyderabad_page'))
        elif selected_location == 'mumbai':
            return redirect(url_for('mumbai_page'))

    return render_template('information.html')

@app.route('/bangalore', methods=['GET'])
def bangalore_page():
    return render_template('BangloreDoctor.html')

@app.route('/chennai', methods=['GET'])
def chennai_page():
    return render_template('ChennaiDoctor.html')


@app.route('/hyderabad', methods=['GET'])
def hyderabad_page():
    return render_template('HydhrabadDoctor.html')


@app.route('/mumbai', methods=['GET'])
def mumbai_page():
    return render_template('MumbaiDoctor.html')


# Define routes for other locations similarly


@app.route('/result', methods=['GET','POST'])
def result():
    if 'imageUpload' not in request.files:
        return 'No file part'
    file = request.files['imageUpload']
    if file.filename == '':
        return 'No selected file'
    file.save(file.filename)
    img = image.load_img(file.filename, target_size=(224, 224)) # Load the image
    x = image.img_to_array(img) # Convert the image to numpy array
    x = np.expand_dims(x, axis=0) # Add a dimension for the batch
    x = x / 255.0 # Normalize
    preds = model.predict(x) # Predict

    # Get the class with highest probability
    preds_class = np.argmax(preds, axis=1)

    # Here you need to map the predicted class to its label
    labels = {0: 'Melanocytic nevi', 1: 'Melanoma', 2: 'Benign keratosis', 3: 'Basal cell carcinoma', 4: 'Actinic keratoses', 5: 'Vascular lesions', 6: 'Dermatofibroma', 7: 'Squamous cell carcinoma'}
    pred_label = labels[preds_class[0]]

    # Calculate the percentage for each class
    preds_percentage = preds[0] * 100
    labels_list = [labels[i] for i in range(len(labels))]

    # Render the result.html template and pass the prediction results
    return render_template('result.html', prediction_label=pred_label, prediction_percentages=preds_percentage.tolist(), labels=labels_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
