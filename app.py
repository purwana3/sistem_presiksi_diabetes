from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os


app = Flask(__name__)

model = load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari form dengan nama field feature0 sampai feature7
        features = []
        for i in range(8):
            field_name = f'feature{i}'
            features.append(float(request.form[field_name]))
        
        # Mengubah list menjadi numpy array dan reshape
        input_data = np.array([features])
        
        # Scaling data
        input_scaled = scaler.transform(input_data)
        
        # Melakukan prediksi
        prediction = model.predict(input_scaled)[0][0]
        
        # Menentukan hasil prediksi
        result = "Pasien kemungkinan MENGIDAP diabetes" if prediction > 0.5 else "Pasien kemungkinan TIDAK mengidap diabetes"
        
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

