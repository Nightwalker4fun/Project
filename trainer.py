# import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_model(df, model_type= "decision_tree"):
    # prepare the data
    x = df [["age", "education", "marital_status"]]
    y = df ["class"]