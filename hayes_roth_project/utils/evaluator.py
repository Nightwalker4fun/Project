from sklearn.metrics import classification_report
import json

def evaluate_model(model, X_test, y_test, save=False, filename='results.json'):
    """Evaluate the model and optionally save the report."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\nEvaluation Report:")
    print(classification_report(y_test, y_pred))

    if save:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
        print(f" Results saved to {filename}")