# 3a. Load the dataset from a CSV file and display the first 10 rows along with basic statistics. And add a menu
import pandas as  pd
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
        print("4. Exit")
        choice = input("Enter your choice: ")
        
        if choice== "1":
            try:
                df = pd.read_csv("data.csv")
                print(" Dataset loaded successfully!")
                print("\n Top 10 rows:")
                print(df.head(10))
                print("\n Basic statistics:")
                print(df.describe(include='all'))
            except FileNotFoundError:
                print("Error: 'data.csv' file not found.")
        elif choice== "2":
            if df is None:
                print("Please load the dataset first.")
            else:
                print("Select model type:")
                print("1. Decision Tree")
                print("2. K-Nearest Neighbors")
                model_choice = input("Enter your choice: ")
                if model_choice== "1":
                    model_type = "decision_tree"
                elif model_choice== "2":
                    model_type= "knn"
                else:
                    print("Invalid choice. Defaulting to Decision Tree.")
                    model_type = "decision_tree"

                model, x_test, y_test = train_model(df, model_type)
                

        elif choice == "3":
            if model is None or x_test is None or y_test is None:
                print("Please train the model first.")
            else: 
                print("Evaluating model performance...")
                y_pred = model.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                report_str = str(report)
                print(f"Accuracy: {acc}")
                print("Classification Report:")
                print(report_str)
                

                #Save performance to file
                with open("model_performance.txt", "w") as f:
                    f.write(f"Accuracy: {acc:.4f}\n\n")
                    f.write(report_str)
                print("Performance saved to 'model_performance.txt")

        elif choice== "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
