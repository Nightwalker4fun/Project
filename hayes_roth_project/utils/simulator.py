def simulate_prediction(model):
    """Simulate a prediction by asking the user for input features."""
    print("\n🔮 Simulate Prediction:")
    try:
        # Adjust these inputs based on your dataset's features
        age = int(input("Enter age: "))
        education = int(input("Enter education level (e.g. 1–3): "))
        marital_status = int(input("Enter marital status (e.g. 1–3): "))

        sample = [[age, education, marital_status]]
        prediction = model.predict(sample)
        print(f" Predicted class: {prediction[0]}")
    except Exception as e:
        print(f" Error during prediction: {e}")