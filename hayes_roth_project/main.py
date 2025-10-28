import pandas as pd
from models.trainer import train_model
from sklearn.metrics import accuracy_score, classification_report

def main():
    df = None
    model = None
    x_test = None
    y_test = None

    while True:
        print("\nMenu:")
        print("1. Load Dataset")
        print("2. Train Model")
        print("3. Evaluate and Save Model Performance")
        print("4. Simulate Real Environment (Predict New Sample)")
        print("5. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            try:
                df = pd.read_csv("data.csv")
                print("Dataset loaded successfully!")
                print("\nTop 10 rows:")
                print(df.head(10))
                print("\nBasic statistics:")
                print(df.describe(include='all'))

                print(df.columns)
            except FileNotFoundError:
                print("Error: 'data.csv' file not found.")
                
        elif choice == "2":
            if df is None:
                print("Please load the dataset first.")
            else:
                print("Select model type:")
                print("1. Decision Tree")
                print("2. K-Nearest Neighbors")
                model_choice = input("Enter your choice: ")
                if model_choice == "1":
                    model_type = "decision_tree"
                elif model_choice == "2":
                    model_type = "knn"
                else:
                    print("Invalid choice. Defaulting to Decision Tree.")
                    model_type = "decision_tree"

                model, x_test, y_test = train_model(df, model_type)
                
        elif choice == "3":
            if model is None or x_test is None or y_test is None:
                print("Please train the model first.")
            else:
                # Ask if user wants to load a specific evaluation file
                load_file = input("Do you want to load a specific file for evaluation? (y/n): ")
                
                if load_file.lower() == 'y':
                    eval_file = input("Enter the evaluation file name: ")
                    try:
                        eval_df = pd.read_csv(eval_file)
                        x_eval = eval_df[["hobby", "age", "education", "marital_status"]]
                        y_eval = eval_df["class"]
                        print(f"Evaluation file '{eval_file}' loaded successfully!")
                    except FileNotFoundError:
                        print(f"Error: '{eval_file}' not found. Using test set from training.")
                        x_eval = x_test
                        y_eval = y_test
                    except Exception as e:
                        print(f"Error loading file: {e}. Using test set from training.")
                        x_eval = x_test
                        y_eval = y_test
                else:
                    print("Using test set from training data (80/20 split).")
                    x_eval = x_test
                    y_eval = y_test
                
                print("\nEvaluating model performance...")
                y_pred = model.predict(x_eval)
                acc = accuracy_score(y_eval, y_pred)
                report = classification_report(y_eval, y_pred)
                report_str = str(report)
                print(f"Accuracy: {acc:.4f}")
                print("Classification Report:")
                print(report_str)
                
                # Ask if user wants to save performance to file
                save_choice = input("\nDo you want to save the performance to a file? (y/n): ")
                if save_choice.lower() == 'y':
                    file_name = input("Enter the filename: ")
                    try:
                        with open(file_name, "w") as f:
                            f.write(f"Accuracy: {acc:.4f}\n\n")
                            f.write(report_str)
                        print(f"Performance saved to '{file_name}'")
                    except Exception as e:
                        print(f"Error saving file: {e}")
                else:
                    print("Results were not saved.")
                    
        elif choice == "4":
            if model is None:
                print("Please train a classification model first.")
            else:
                print("\n--- Predict New Sample ---")
                print("Enter the attribute values for the new sample:")
                print("(Hayes-Roth dataset: hobby, age, education, marital status)")
                
                try:
                    # Get input for each feature
                    hobby = int(input("Enter hobby (1-3): "))
                    age = int(input("Enter age (1-4): "))
                    education = int(input("Enter education (1-4): "))
                    marital_status = int(input("Enter marital status (1-4): "))
                    
                    # Validate ranges
                    if not (1 <= hobby <= 3):
                        print("Warning: Hobby should be between 1-3")
                    if not (1 <= age <= 4):
                        print("Warning: Age should be between 1-4")
                    if not (1 <= education <= 4):
                        print("Warning: Education should be between 1-4")
                    if not (1 <= marital_status <= 4):
                        print("Warning: Marital status should be between 1-4")
                    
                    # Create a DataFrame with the same structure as training data
                    new_sample = pd.DataFrame({
                        'hobby': [hobby],
                        'age': [age],
                        'education': [education],
                        'marital_status': [marital_status]
                    })
                    
                    # Make prediction
                    prediction = model.predict(new_sample)
                    print(f"\nPredicted class: {prediction[0]}")
                    
                except ValueError as ve:
                    print(f"Error: Please enter valid integer values. Details: {ve}")
                except Exception as e:
                    print(f"Error making prediction: {e}")
                    print("Hint: Check that your CSV column names match exactly what the model was trained on.")

        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()