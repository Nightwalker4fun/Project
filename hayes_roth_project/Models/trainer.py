# import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_model(df, model_type= "decision_tree"):
    # prepare the data
    x = df [["age", "education", "marital_status"]]
    y = df ["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)
    
    if model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "knn":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(x_train, y_train)
    print(f"{model_type} model trained successfully!")
    return model, x_test, y_test
