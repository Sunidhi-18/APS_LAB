from sklearn.metrics import confusion_matrix
from cancerdata import load_and_preprocess_data, get_train_test_data
from cancermean import train_model

def evaluate_matrix(model, x_test, y_test):
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return cm

if __name__ == "__main__":
    x, y, _ = load_and_preprocess_data()
    x_train, x_test, y_train, y_test = get_train_test_data(x, y)
    
    model = train_model(x_train, y_train)
    evaluate_matrix(model, x_test, y_test)