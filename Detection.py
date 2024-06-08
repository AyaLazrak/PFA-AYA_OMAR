import pickle
from generalize import CreditCardFraudDetection
from AI import AIModel  # Replace 'your_ai_model' with the name of your AI model class
from sklearn.svm import SVC
import json
class FraudDetectionAI:
    def __init__(self, data_file):
        self.data_file = data_file
        self.classifier = None
        self.ai_model = None

    def train(self):
        # Step 1: Instantiate CreditCardFraudDetection
        self.classifier = CreditCardFraudDetection(self.data_file)

        # Step 2: Execute data loading, exploration, preprocessing, splitting, and training
        self.classifier.run()
        print('end of classifier part 1')
        # Step 3: Instantiate and train the AIModel
        model = SVC()  # Choose your desired model, such as SVC from scikit-learn
        print('start svc')
        
        self.ai_model = AIModel(model)
        print( 'start training ')
        self.ai_model.train(self.classifier.X_train, self.classifier.y_train)
        print('end training')
    def evaluate(self):
        # Evaluate the trained AI model using the test dataset
        accuracy, precision, recall, f1 = self.ai_model.evaluate(self.classifier.X_test, self.classifier.y_test)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")

    def predict(self, data):
        # Use the trained AI model for prediction on new data
        predictions = self.ai_model.predict(data)
        return predictions

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            model_data = {
                'classifier': self.classifier,
                'ai_model': self.ai_model
            }
            pickle.dump(model_data, f)
    def save_modeltxt(self, filename):
        with open(filename, 'w') as f:
            model_data = {
                'classifier': self.classifier,
                'ai_model': self.ai_model
            }
            json.dump(model_data, f)

def convert_binary_to_text(binary_file, text_file):
    with open(binary_file, 'rb') as f_in:
        # Load the binary data
        data = pickle.load(f_in)

    with open(text_file, 'w') as f_out:
        # Write the data as text
        f_out.write(str(data))
         
         
# Example usage
data_file = 'C:/Users/ayala/Desktop/Front-Pfa1/Front-Pfa/treatement/creditcard.csv'  # Provide the path to your data file
fraud_detection = FraudDetectionAI(data_file)
fraud_detection.train()
fraud_detection.evaluate()
fraud_detection.save_model('model.pkl')
#fraud_detection.save_modeltxt('result.txt')
#convert_binary_to_text('model.pkl','result.txt')