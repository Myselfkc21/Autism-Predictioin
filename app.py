from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
#USING THIS ONE

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Correct the column name 'austim' to 'autism' in both train and test data
train_data.rename(columns={'austim': 'autism'}, inplace=True)
test_data.rename(columns={'austim': 'autism'}, inplace=True)

# Drop the ID column
train_data = train_data.drop(columns=['ID'])
test_data = test_data.drop(columns=['ID'])

# Encode categorical variables with handling unseen labels
label_encoders = {}
for column in ['gender', 'ethnicity', 'jaundice', 'autism', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column].astype(str))
    if column in test_data.columns:
        test_data[column] = test_data[column].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    label_encoders[column] = le

# Separate features and target
X = train_data.drop(columns=['Class/ASD']) #independent var
y = train_data['Class/ASD']        #dependent var

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
svc_model = SVC(kernel='linear', random_state=42)

# Fit the model
svc_model.fit(X_train, y_train)

# Validate the model
y_pred = svc_model.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_pred)
classification_report_result = classification_report(y_val, y_pred)

print("Validation Accuracy:", validation_accuracy)
print("Classification Report:\n", classification_report_result)

# Save the model
joblib.dump(svc_model, 'svc_model.pkl')

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')


app = Flask(__name__)

# Load the model and label encoders
svc_model = joblib.load('svc_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to preprocess the input data
def preprocess_input(input_data, label_encoders):
    for column, le in label_encoders.items():
        if input_data[column] in le.classes_:
            input_data[column] = le.transform([input_data[column]])[0]
        else:
            input_data[column] = -1
    return input_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            'A1_Score': int(request.form['A1_Score']),
            'A2_Score': int(request.form['A2_Score']),
            'A3_Score': int(request.form['A3_Score']),
            'A4_Score': int(request.form['A4_Score']),
            'A5_Score': int(request.form['A5_Score']),
            'A6_Score': int(request.form['A6_Score']),
            'A7_Score': int(request.form['A7_Score']),
            'A8_Score': int(request.form['A8_Score']),
            'A9_Score': int(request.form['A9_Score']),
            'A10_Score': int(request.form['A10_Score']),
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jaundice': request.form['jaundice'],
            'autism': request.form['autism'],
            'contry_of_res': request.form['contry_of_res'],
            'used_app_before': request.form['used_app_before'],
            'result': float(request.form['result']),
            'age_desc': request.form['age_desc'],
            'relation': request.form['relation']
        }

        # Preprocess the input data
        input_df = pd.DataFrame([input_data])
        input_df = input_df.apply(preprocess_input, axis=1, label_encoders=label_encoders)

        # Make prediction
        prediction = svc_model.predict(input_df)

        # Output the result
        if prediction[0] == 1:
            result = "The person is predicted to be autistic."
        else:
            result = "The person is predicted to be non-autistic."

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
