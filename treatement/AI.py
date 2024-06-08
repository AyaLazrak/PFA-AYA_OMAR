from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AIModel:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        return accuracy, precision, recall, f1
    def train_model(self, model):
        ai_model = AIModel(model)
        ai_model.train(self.X_train, self.y_train)
        return ai_model

    def evaluate_model(self, model):
        accuracy, precision, recall, f1 = model.evaluate(self.X_test, self.y_test)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
