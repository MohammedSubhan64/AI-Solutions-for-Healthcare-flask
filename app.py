from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


with open('models/stroke_my_model.pkl', 'rb') as f:
# with open('C:\Users\Subhan\Documents\Health Care Project\flask code\models\stroke_my_model.pkl', 'rb') as f:
    stroke2_model = pickle.load(f)

# @app.route('/stroke2')
# def stroke2():
#     return render_template('stroke2.html', prediction_result=None)

@app.route('/stroke2', methods=['POST'])
def stroke2():
    # prediction_result = None
    try:
        # Retrieve form inputs from the HTML form
        patient_id = int(request.form['PATIENT_ID'])
        visit_id = int(request.form['VISIT_ID'])
        clinic_code = int(request.form['CLINIC_CODE'])
        race = int(request.form['RACE'])
        discharge_type = int(request.form['DISCHARGE_TYPE'])
        facility = int(request.form['FACILITY'])
        visit_status = int(request.form['VISIT_STATUS'])
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create an array with the feature values
        input_data = np.array([[
            patient_id,
            visit_id,
            clinic_code,
            race,
            discharge_type,
            facility,
            visit_status,
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            avg_glucose_level,
            bmi,
            smoking_status
        ]])

        # Make a prediction using the trained model
        prediction = stroke2_model.predict(input_data)

        # Return the result based on the prediction
        if prediction == 1:
            prediction_result = "The patient is at high risk of having a stroke."
        else:
            prediction_result = "The patient is not at high risk of having a stroke."
        
        return render_template('stroke2.html', prediction_result=prediction_result)

    except Exception as e:
        # Handle exceptions (e.g., missing values or incorrect data types)
        return render_template('stroke2.html', prediction_result="Error in prediction. Please check the input values.")



# Load the pre-trained model (make sure the model.pkl file is in the same directory as your app.py)
kidney_model = None

with open("models/kidney1_model.pkl", 'rb') as f:
    kidney_model = pickle.load(f)

# Define the route for the home page (where the form will be displayed)
@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    prediction_result = None

    if request.method == "POST":
        # Retrieve form data
        features = [
            float(request.form["age"]),
            float(request.form["bp"]),
            float(request.form["sg"]),
            float(request.form["al"]),
            float(request.form["su"]),
            float(request.form["bgr"]),
            float(request.form["bu"]),
            float(request.form["sc"]),
            float(request.form["sod"]),
            float(request.form["pot"]),
            float(request.form["hemo"]),
            int(request.form["rbc_n"]),
            int(request.form["pc_n"]),
            int(request.form["pcc_n"]),
            int(request.form["ba_n"]),
            int(request.form["pcv_n"]),
            int(request.form["wc_n"]),
            int(request.form["rc_n"]),
            int(request.form["htn_n"]),
            int(request.form["dm_n"]),
            int(request.form["cad_n"]),
            int(request.form["appet_n"]),
            int(request.form["pe_n"]),
            int(request.form["ane_n"]),
        ]
        
        # Convert the features to a numpy array (if needed for your model)
        features = np.array(features).reshape(1, -1)  # Reshaping for single prediction
        
        # Make prediction using the model
        prediction = kidney_model.predict(features)

        # Convert prediction to human-readable format
        if prediction == 1:
            prediction_result = "Risk of Kidney Disease: High"
        else:
            prediction_result = "Risk of Kidney Disease: Low"
    
    return render_template("kidney.html", prediction_result=prediction_result)

import joblib

# Load the pre-trained model (assuming the model is saved as breast_cancer_model.pkl)
with open('models/breast_model.pkl', 'rb') as model_file:
    breast_model = joblib.load(model_file)

@app.route('/breast', methods=['GET', 'POST'])
def breast():
    prediction_result = None

    if request.method == 'POST':
        # Get input values from the form and convert to float as necessary
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
        
        # Create input data array for prediction
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                               smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
                               symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se,
                               area_se, smoothness_se, compactness_se, concavity_se, concave_points_se,
                               symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
                               area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                               symmetry_worst, fractal_dimension_worst]])

       # Reshape the feature array as expected by the model (1, 30)
        # input_data = input_data.reshape(1, -1)
        # Predict using the model
        prediction = breast_model.predict(input_data)
        
        # Interpret the prediction (assuming 0 = Benign, 1 = Malignant)
        if prediction == 1:
            prediction_result = "Malignant (Cancerous)"
        else:
            prediction_result = "Benign (Non-cancerous)"
    
    # Render the HTML template with the prediction result
    return render_template('breast.html', prediction_result=prediction_result)


# Load the trained model (no scaler loading)
with open('models/heart1_disease_model.pkl', 'rb') as model_file:
    heart_model = pickle.load(model_file)


# Define the route for prediction
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    try:
        # Extract the input values from the form
        features = [int(request.form.get('age')),
                    int(request.form.get('sex')),
                    int(request.form.get('cp')),
                    int(request.form.get('trestbps')),
                    int(request.form.get('chol')),
                    int(request.form.get('fbs')),
                    int(request.form.get('restecg')),
                    int(request.form.get('thalach')),
                    int(request.form.get('exang')),
                    float(request.form.get('oldpeak')),
                    int(request.form.get('slope')),
                    int(request.form.get('ca')),
                    int(request.form.get('thal'))]

        # Convert to numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)

        # Make the prediction (no scaling applied)
        prediction = heart_model.predict(features)

        # Return the result
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        return render_template('heart.html', prediction_result=result)

    except Exception as e:
        return render_template('heart.html', prediction_result="Error: " + str(e))


# Load the pre-trained model
with open('models/stroke1_model.pkl', 'rb') as model_file:
    stroke_model = pickle.load(model_file)

import pandas as pd
@app.route('/stroke', methods=["GET",'POST'])
def stroke():
    # Get values from the form
    try:
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease']) 
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        gender_n = int(request.form['gender_n'])
        work_type_n = int(request.form['work_type_n'])
        ever_married_n = int(request.form['ever_married_n'])
        smoking_status_n = int(request.form['smoking_status_n'])
        Residence_type_n = int(request.form['Residence_type_n'])

        # Create a DataFrame from the input values
        input_data = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose_level, bmi, 
                                    gender_n, work_type_n, ever_married_n, smoking_status_n, Residence_type_n]],
                                  columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 
                                           'gender_n', 'work_type_n', 'ever_married_n', 'smoking_status_n', 'Residence_type_n'])
        # Convert to numpy array and reshape for prediction
        # features = np.array(input_data).reshape(1, -1)

        # Make the prediction (no scaling applied)
        # prediction = stroke_model.predict(features)
        # Make prediction using the model
        prediction = stroke_model.predict(input_data)

        # Convert the result to a human-readable format
        if prediction == 1:
            prediction_result = 'High Risk of Heart Disease'
        else:
            prediction_result = 'Low Risk of Heart Disease'
        return render_template('stroke.html', prediction_result=prediction_result)
    except Exception as e:
        result = f"Error occurred: {e}"

    return render_template('stroke.html', prediction_result=result)




# Load the trained model
liver_model = pickle.load(open('models/liver_model.pkl', 'rb'))


@app.route('/liver', methods=['GET','POST'])
def liver():
    try:
        # Collect form data
        age = int(request.form['age'])
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphatase = int(request.form['alkaline_phosphatase'])
        alamine_aminotransferase = int(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = int(request.form['aspartate_aminotransferase'])
        total_proteins = float(request.form['total_proteins'])
        albumin = float(request.form['albumin'])
        albumin_globulin_ratio = float(request.form['albumin_globulin_ratio'])
        gender_n = int(request.form['gender_n'])

        # Prepare input features as a numpy array
        input_features = np.array([[age, total_bilirubin, direct_bilirubin, alkaline_phosphatase,
                                    alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                                    albumin, albumin_globulin_ratio, gender_n]])

        # Get prediction from the model
        prediction = liver_model.predict(input_features)

        # Map the prediction result to a readable format
        result = 'Liver Disease Detected' if prediction[0] == 1 else 'No Liver Disease'

        return render_template('liver.html', prediction_result=result)
    
    except Exception as e:
        prediction_result = f"Error occurred: {e}"
        return render_template('liver.html', prediction_result=prediction_result)




# from sklearn.preprocessing import StandardScaler


# Load the trained model
diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        try:
            # Collect input from the form
            pregnancies = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            blood_pressure = int(request.form['blood_pressure'])
            skin_thickness = int(request.form['skin_thickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
            age = int(request.form['age'])

            # Create the input array for prediction
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree_function, age]])

            # Normalize the input data using the same scaler as the model's training
            # (assuming the model requires scaling; if not, you can remove this part)
            # scaler = StandardScaler()
            # input_data_scaled = scaler.fit_transform(input_data)

            # Make a prediction using the model
            prediction = diabetes_model.predict(input_data)

            # Return the result
            prediction_result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        
        except ValueError as e:
            # Handle value errors (e.g., invalid input types like strings instead of numbers)
            error_message = f"Invalid input: {e}. Please make sure all fields are filled correctly."
        
        except Exception as e:
            # Catch other exceptions and show a general error message
            error_message = f"An error occurred: {e}. Please try again later."

    return render_template('diabetes_with_bmi.html', prediction_result=prediction_result, error_message=error_message)



# To disable CSRF protection temporarily, add the following to your Flask app:
# app.config['WTF_CSRF_ENABLED'] = False

if __name__ == '__main__':
    app.run(debug=True,port=8080)
