import pandas as pd
from models.trainer import train_model
from utils.evaluator import evaluate_model
from utils.simulator import simulate_prediction

def main():
    df = None
    model = None
    X_test = None
    y_test = None

    while True:
        print("\n📋 Menu:")
        print("1. Load dataset")
        print("2. Train model")
        print("3. Evaluate model")
        print("4. Simulate prediction")
        print("5. Exit")

        choice = input("Enter your choice (1–5): ")

        if choice == "1":
            # Load dataset
            try:
                df = pd.read_csv("data.csv")
                print("✅ Dataset loaded successfully!")
                print("\n📌 Top 10 rows:")
                print(df.head(10))
                print("\n📊 Basic statistics:")
                print(df.describe(include='all'))
            except FileNotFoundError:
                print("❌ Error: 'data.csv' file not found.")

        elif choice == "2":
            # Train model
            if df is None:
                print("⚠️ Please load the dataset first.")
            else:
                print("🧠 Choose model type:")
                print("1. Decision Tree")
                print("2. K-Nearest Neighbors")
                model_choice = input("Enter 1 or 2: ")
        elif c

                if model_choice == "1":
                    model_type = "decision_tree"
                elif model_choice == "2":
                    model_type = "knn"
                else:
                    print("❌ Invalid choice. Defaulting to Decision Tree.")
                    model_type = "decision_tree"

                model, X_test, y_test = train_model(df, model_type)

        elif choice == "3":
            # Evaluate model
            if model is None or X_test is None or y_test is None:
                print("⚠️ You must train a model first.")
            else:
                save = input("Do you want to save the results? (yes/no): ").lower() == "yes"
                filename = input("Enter filename (e.g. results.json): ") if save else "results.json"
                evaluate_model(model, X_test, y_test, save, filename)

        elif choice == "4":
            # Simulate prediction
            if model is None:
                print("⚠️ You must train a model first.")
            else:
                simulate_prediction(model)

        elif choice == "5":
            print("👋 Exiting program.")
            break

        else:
            print("❌ Invalid choice. Please select 1–5.")

if __name__ == "__main__":
    main()