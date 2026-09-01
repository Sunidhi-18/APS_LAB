from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model(x_train, y_train):
    #Train a logistic regression model
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    # Fit the pipeline
    pipeline.fit(x_train, y_train)
    return pipeline