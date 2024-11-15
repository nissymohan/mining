from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Step 1: Load the trained RandomForest model and label encoders
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

with open('label_encoders2.pkl', 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)

with open('country_onehot_encoder2.pkl', 'rb') as f:
    country_onehot_encoder = pickle.load(f)

# Load the pre-trained models
with open('attraction_models2.pkl', 'rb') as f:
    models = pickle.load(f)

    #print(label_encoders)


with open('X.pkl', 'rb') as col_file:
    X = pickle.load(col_file)

#print(f"Loaded columns: {X_columns}")

# Step 2: Define the prediction function
def predict_top2_attraction_types(model, visit_mode, visit_month, country, label_encoders, X):
    # Step 1: Encode 'visit_mode' and 'country' using the saved LabelEncoders
    if visit_mode not in label_encoders['VisitMode'].classes_:
        raise ValueError(f"Invalid visit mode: {visit_mode}")
    if country not in label_encoders['Country'].classes_:
        raise ValueError(f"Invalid country: {country}")

    visit_mode_encoded = label_encoders['VisitMode'].transform([visit_mode])[0]
    country_encoded = label_encoders['Country'].transform([country])[0]

    # Step 2: Generate one-hot encoding for 'Country'
    country_one_hot = [0] * len(label_encoders['Country'].classes_)
    country_one_hot[country_encoded] = 1

    # Step 3: Generate interaction feature 'VisitMode_Country'
    visit_mode_country_interaction = [visit_mode_encoded * val for val in country_one_hot]

    # Step 4: Create the interaction feature 'VisitMode_VisitMonth'
    visit_mode_visit_month_interaction = visit_mode_encoded * visit_month

    # Step 5: Assemble the feature vector for prediction
    input_data = pd.DataFrame([[
        visit_mode_encoded,           # VisitMode
        visit_month,                  # VisitMonth
        visit_mode_visit_month_interaction, # VisitMode_VisitMonth
        *country_one_hot,             # Country one-hot encoding
        *visit_mode_country_interaction # VisitMode_Country interaction
    ]], columns=[
        'VisitMode', 'VisitMonth', 'VisitMode_VisitMonth'
    ] + [f'Country_{i}' for i in range(len(country_one_hot))] +
        [f'VisitMode_Country_{i}' for i in range(len(visit_mode_country_interaction))]
    )

    # Step 6: Ensure columns are in the same order as during training
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Step 7: Predict the probabilities for each class
    probabilities = model.predict_proba(input_data)[0]

    # Step 8: Get the indices of the top 2 highest probabilities
    top2_indices = np.argsort(probabilities)[-2:][::-1]  # Get top 2 indices sorted in descending order

    # Step 9: Convert indices to actual labels
    top2_attractions = label_encoders['AttractionType'].inverse_transform(top2_indices)

    # Print the top 2 predictions
    print(f"Top 1 Prediction: {top2_attractions[0]}")
    print(f"Top 2 Prediction: {top2_attractions[1]}")
    print(top2_attractions.dtype)
    return top2_attractions

def predict_top3_attraction_types(country, visit_mode, visit_month, attraction_type):
    """
    Predict the top-3 attractions based on the given inputs.
    """
    # Step 1: Encode the inputs using the loaded encoders
    country_encoded = encoders['Country'].transform([country])[0]
    visit_mode_encoded = encoders['VisitMode'].transform([visit_mode])[0]
    attraction_type_encoded = encoders['AttractionType'].transform([attraction_type])[0]

    # Step 2: One-hot encode the country using the saved OneHotEncoder
    # Ensure input is a DataFrame with the correct column name
    country_df = pd.DataFrame([[country_encoded]], columns=['Country'])
    country_onehot = country_onehot_encoder.transform(country_df).flatten()

    # Step 3: Generate the interaction features
    visit_mode_visit_month = visit_mode_encoded * visit_month
    visit_mode_country = visit_mode_encoded * country_onehot

    # Step 4: Prepare the input feature vector
    input_features = np.hstack([
        country_onehot,
        [visit_mode_encoded],
        [visit_month],
        [visit_mode_visit_month],
        visit_mode_country
    ]).reshape(1, -1)

    # Step 5: Load the model corresponding to the attraction type
    model = models.get(attraction_type_encoded)
    if model is None:
        raise ValueError(f"No model found for AttractionType '{attraction_type}'")

    # Step 6: Make predictions
    predictions = model.predict_proba(input_features)

    # Get the top-3 predictions based on probabilities
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_attraction_encoded = model.classes_[top_3_indices]

    # Step 7: Inverse transform the encoded attraction labels to get the original names
    top_3_attractions = encoders['Attraction'].inverse_transform(top_3_attraction_encoded)

    return top_3_attractions


# Load the saved models, label encoders, and attractions dictionary
with open('models3.pkl', 'rb') as f:
    models3 = pickle.load(f)

with open('label_encoders3.pkl', 'rb') as f:
    label_encoders3 = pickle.load(f)

with open('attractions_dict3.pkl', 'rb') as f:
    attractions_dict3 = pickle.load(f)

# Function to predict the top 3 attractions
def predict_top_3_attractions(country, visit_mode, attraction_type):
    """
    Predicts the top 3 attractions based on the input country, visit mode, and attraction type.
    """
    # Encode input features using the existing label encoders
    try:
        encoded_country = label_encoders3['Country'].transform([country])[0]
        encoded_visit_mode = label_encoders3['VisitMode'].transform([visit_mode])[0]
        encoded_attraction_type = label_encoders3['AttractionType'].transform([attraction_type])[0]
    except ValueError as e:
        print(f"Error in encoding: {e}")
        return []

    # Prepare input data
    input_df = pd.DataFrame([[encoded_country, encoded_visit_mode, encoded_attraction_type]],
                            columns=['Country', 'VisitMode', 'AttractionType'])

    # Get the model for the specified AttractionType
    if encoded_attraction_type not in models3:
        print("No model found for the specified AttractionType.")
        return []

    model = models3[encoded_attraction_type]
    probabilities = model.predict_proba(input_df)[0]
    
    # Get the classes corresponding to these probabilities
    model_classes = model.classes_

    # Get the attractions for the given AttractionType from the preprocessed dictionary
    attractions_of_type = attractions_dict3.get(encoded_attraction_type, [])
    
    # Map the encoded attractions to the indices in the model's classes
    relevant_indices = [np.where(model_classes == attraction)[0][0] for attraction in attractions_of_type if attraction in model_classes]

    # Extract probabilities for the relevant attractions
    relevant_probs = [probabilities[i] for i in relevant_indices]
    
    # Sort the relevant attractions by their probabilities in descending order
    top_3_indices = np.argsort(relevant_probs)[-3:][::-1]
    top_3_encoded_attractions = [attractions_of_type[i] for i in top_3_indices]

    # Decode the top 3 attractions to their original names
    top_3_attractions = label_encoders3['Attraction'].inverse_transform(top_3_encoded_attractions)
    return top_3_attractions





def predict_top3_visit_months(country, visit_mode, attraction_type, attraction):

    # Load the saved label encoders
    with open('label_encode-task3.pkl', 'rb') as f:
        encoders3 = pickle.load(f)

    # Load the one-hot encoder for country
    with open('onehot_encode_country-task3.pkl', 'rb') as f:
        country_onehot_encoder3 = pickle.load(f)

    # Load the pre-trained models
    with open('random_forest_models_task3.pkl', 'rb') as f:
        models3_2 = pickle.load(f)
    
    # Step 1: Encode the input features using the label encoders
    encoded_country = encoders3['Country'].transform([country])[0]
    encoded_visit_mode = encoders3['VisitMode'].transform([visit_mode])[0]
    encoded_attraction_type = encoders3['AttractionType'].transform([attraction_type])[0]
    encoded_attraction = encoders3['Attraction'].transform([attraction])[0]

    # Step 2: One-hot encode the country (ensure it's a DataFrame)
    country_onehot = country_onehot_encoder3.transform(pd.DataFrame([[encoded_country]], columns=['Country'])).flatten()

    # Step 3: Create interaction features
    visit_mode_attraction_type = encoded_visit_mode * encoded_attraction_type
    visit_mode_country = encoded_visit_mode * country_onehot

    # Step 4: Prepare the feature vector for prediction
    X = np.hstack([
        country_onehot,
        [encoded_visit_mode],
        [encoded_attraction_type],
        [visit_mode_attraction_type],
        visit_mode_country
    ]).reshape(1, -1)  # Reshape to match the expected input format

    # Step 5: Predict using the pre-trained model for the given attraction
    if encoded_attraction in models3_2:
        model = models3_2[encoded_attraction]
        predicted_months = model.predict_proba(X)[0]

        # Get the top-3 months with the highest probabilities
        top3_indices = np.argsort(predicted_months)[-3:][::-1]  # Sort in descending order
        top3_months = [i + 1 for i in top3_indices]  # Convert indices to months (1-12)

        #print(f"Top-3 visit months for '{attraction}' are: {top3_months}")
        return top3_months
    else:
        print(f"No model found for attraction: '{attraction}'")
        return []
    


# Route to serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'min685.html')

# Step 3: Define the Flask route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    visit_mode = data['visit_mode']
    visit_month = data['visit_month']
    country = data['country']
    
    try:
        # Make predictions using the model
        top2_predictions = predict_top2_attraction_types(
            model, visit_mode, visit_month, country, label_encoders, X
        )
        top2_predictions = [str(prediction) for prediction in top2_predictions]
        return jsonify({'predicted_attraction_types': top2_predictions})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500
    

@app.route('/predict_attraction_place', methods=['POST'])
def predict_attraction_place():
    data = request.json
    visit_mode = data['visit_mode']
    visit_month = data['visit_month']
    country = data['country']
    attraction_type = data['attraction_type']
    
    try:
        # Make predictions using the model
        top3_predictions = predict_top3_attraction_types(
            country, visit_mode, visit_month, attraction_type
        )
        top3_predictions = [str(prediction) for prediction in top3_predictions]
        print(top3_predictions)
        return jsonify({'predicted_attraction_places': top3_predictions})
    except ValueError as e:
        print("Hello There")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print("Hello Here")
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500
    
@app.route('/predict_attraction_place3', methods=['POST'])
def predict_attraction():
    data = request.json
    visit_mode = data['visit_mode']
    attraction_type = data['attraction_type']
    country = data['country']
    
    try:
        # Make predictions using the model
        top_3_attractions = predict_top_3_attractions(country, visit_mode, attraction_type)
        top3_predictions = [str(prediction) for prediction in top_3_attractions]
        return jsonify({'predicted_attraction_places2': top3_predictions})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500
    

@app.route('/predict_best_months', methods=['POST'])
def predict_months():
    data = request.json
    visit_mode = data['visit_mode']
    attraction_type = data['attraction_type']
    country = data['country']
    attraction =data['attraction_place']
    
    try:
        # Make predictions using the model
       top3_months = predict_top3_visit_months(country, visit_mode, attraction_type, attraction)
       top_months = [str(prediction) for prediction in top3_months]
       return jsonify({'best_months': top_months})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
 