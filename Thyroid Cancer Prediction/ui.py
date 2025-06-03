import gradio as gr 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('D:\Download\Thyroid Cancer Prediction\Thyroid_Diff.csv')

X = data.drop(columns=['Recurred'])  
y = data['Recurred'] 

categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
svc = SVC()
rf = RandomForestClassifier()

models = {'Logistic Regression': logreg, 'SVM': svc, 'Random Forest': rf}
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

def prepare_input_for_prediction(input_data):
    input_df = pd.DataFrame([input_data])  
    input_encoded = pd.get_dummies(input_df)
    
    missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0  

    input_encoded = input_encoded[X_encoded.columns]
 
    input_scaled = scaler.transform(input_encoded)
    
    return input_scaled

def predict(age, gender, smoking, hx_smoking, hx_radiotherapy, thyroid_function, physical_exam, adenopathy, pathology, focality, risk, t, n, m, stage, response):
    input_data = {
        'Age': age,
        'Gender': gender,
        'Smoking': smoking,
        'Hx Smoking': hx_smoking,
        'Hx Radiotherapy': hx_radiotherapy,
        'Thyroid Function': thyroid_function,
        'Physical Examination': physical_exam,
        'Adenopathy': adenopathy,
        'Pathology': pathology,
        'Focality': focality,
        'Risk': risk,
        'T': t,
        'N': n,
        'M': m,
        'Stage': stage,
        'Response': response
    }
    
    prepared_input = prepare_input_for_prediction(input_data)
    predicted_outcome = best_model.predict(prepared_input)
    predicted_label = label_encoder.inverse_transform(predicted_outcome)  
    
    return predicted_label[0]
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 120, value=30, label="Age"),
        gr.Radio(["M", "F"], label="Gender"),
        gr.Radio(["Yes", "No"], label="Smoking"),
        gr.Radio(["Yes", "No"], label="Hx Smoking"),
        gr.Radio(["Yes", "No"], label="Hx Radiotherapy"),
        gr.Radio(["Euthyroid", "Hyperthyroid", "Hypothyroid"], label="Thyroid Function"),
        gr.Textbox(label="Physical Examination"),
        gr.Radio(["Yes", "No"], label="Adenopathy"),
        gr.Textbox(label="Pathology (e.g., Micropapillary)"),
        gr.Radio(["Uni-Focal", "Multi-Focal"], label="Focality"),
        gr.Radio(["Low", "High"], label="Risk"),
        gr.Dropdown(["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"], label="T staging"),
        gr.Dropdown(["N0", "N1"], label="N staging"),
        gr.Dropdown(["M0", "M1"], label="M staging"),
        gr.Dropdown(["I", "II", "III", "IV"], label="Stage"),
        gr.Dropdown(["Excellent", "Indeterminate", "Incomplete"], label="Response")
    ],
    outputs=gr.Textbox(label="Predicted Outcome"),
    title="Thyroid Cancer Recurrence Prediction",
    description="Enter patient details to predict Recurrence of thyroid cancer."
)

interface.launch(share=True)
