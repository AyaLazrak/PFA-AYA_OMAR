# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# class TransactionProcessor:
#     def __init__(self):
#         self.data = None
#         self.processed_data = None
#         self.classifier = None

#     def train_classifier(self):
#         features = self.processed_data.drop(['Class'], axis=1)
#         labels = self.processed_data['Class']
#         X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#         self.classifier = SVC()
#         self.classifier.fit(X_train, y_train)
#         y_pred = self.classifier.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print("Classifier Accuracy:", accuracy)
#     def load_data(self, file_name):
#         self.data = pd.read_csv(file_name)

#     def preprocess_data(self):
#         # Remove any missing values
#         self.data.dropna(inplace=True)

#         # Scale the features
#         scaler = StandardScaler()
#         self.data['Amount'] = scaler.fit_transform(self.data['Amount'].values.reshape(-1, 1))

#     def visualize_data(self):
#         # Visualize the class distribution
#         class_counts = self.data['Class'].value_counts()
#         class_counts.plot(kind='bar')
#         plt.xlabel('Class')
#         plt.ylabel('Count')
#         plt.title('Class Distribution')
#         plt.show()

#     def remove_outliers(self):
#         # Remove outliers using a threshold
#         threshold = 2.5
#         z_scores = np.abs(self.data['Amount'])
#         self.data = self.data[z_scores < threshold]

#     def reduce_dimensionality(self):
#         # Perform dimensionality reduction using PCA
#         features = self.data.drop(['Class'], axis=1)
#         pca = PCA(n_components=2)
#         reduced_features = pca.fit_transform(features)
#         self.processed_data = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])
#         self.processed_data['Class'] = self.data['Class']

#     def process_transactions(self, file_name):
#         self.load_data(file_name)
#         self.preprocess_data()
#         self.visualize_data()
#         self.remove_outliers()
#         self.reduce_dimensionality()
#         print("Processed Data:")
#         print(self.processed_data.head())


# # Example usage:
# #transaction_processor = TransactionProcessor()
# #transaction_processor.process_transactions('C:\\Users\\hp\\Desktop\\pfa\\bank-data-example-master\\bank_data\\treatement\\creditcard.csv')
# #processed_data = transaction_processor.processed_data

import os

# Définir le nombre de cœurs que vous souhaitez utiliser
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Par exemple, pour utiliser 4 cœurs

# Importer le reste de votre code ici
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix



def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
        alpha=0.1, color="r"
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
        alpha=0.1, color="g"
    )
    plt.plot(
        train_sizes, train_scores_mean, 'o-', color="r", label="Training score"
    )
    plt.plot(
        train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return plt


class CreditCardFraudDetection:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def explore_data(self):
        print(self.df.describe())
        print(self.df.isnull().sum().max())
        print(self.df.columns)
        print('No Frauds', round(self.df['Class'].value_counts()[0] / len(self.df) * 100, 2), '% of the dataset')
        print('Frauds', round(self.df['Class'].value_counts()[1] / len(self.df) * 100, 2), '% of the dataset')
        
        colors = ["#0101DF", "#DF0101"]
        sns.countplot(x=self.df['Class'], palette=colors)
        plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
        plt.show()
        
        fig, ax = plt.subplots(1, 2, figsize=(18,4))
        amount_val = self.df['Amount'].values
        time_val = self.df['Time'].values
        sns.distplot(amount_val, ax=ax[0], color='r')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlim([min(amount_val), max(amount_val)])
        sns.distplot(time_val, ax=ax[1], color='b')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlim([min(time_val), max(time_val)])
        plt.show()
    
    def preprocess_data(self):
        std_scaler = StandardScaler()
        rob_scaler = RobustScaler()

        self.df['scaled_amount'] = rob_scaler.fit_transform(self.df['Amount'].values.reshape(-1, 1))
        self.df['scaled_time'] = rob_scaler.fit_transform(self.df['Time'].values.reshape(-1, 1))

        self.df.drop(['Time', 'Amount'], axis=1, inplace=True)

        scaled_amount = self.df['scaled_amount']
        scaled_time = self.df['scaled_time']

        self.df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        self.df.insert(0, 'scaled_amount', scaled_amount)
        self.df.insert(1, 'scaled_time', scaled_time)

        return self.df
    
    def split_data(self):
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(X, y):
            original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
            original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

        original_Xtrain = original_Xtrain.values
        original_Xtest = original_Xtest.values
        original_ytrain = original_ytrain.values
        original_ytest = original_ytest.values

        return original_Xtrain, original_Xtest, original_ytrain, original_ytest
    
    def train_model(self, model):
        model.fit(self.X_train, self.y_train)
        return model
    
    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
    
    def visualize_learning_curve(self, model):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        plot_learning_curve(model, 'Learning Curve', self.X_train, self.y_train, cv=cv)
        plt.show()
    
    def plot_confusion_matrix(self, model):
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def plot_roc_curve(self, model):
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    def plot_precision_recall_curve(self, model):
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        average_precision = average_precision_score(self.y_test, y_pred_proba)

        plt.plot(recall, precision, label='Avg Precision = %0.2f' % average_precision)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.show()
    
    def visualize_tsne(self):
        data_subset = self.df.sample(n=10000)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(data_subset.drop('Class', axis=1).values)

        data_subset['tsne-2d-one'] = tsne_results[:, 0]
        data_subset['tsne-2d-two'] = tsne_results[:, 1]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="Class",
            palette=sns.color_palette("hsv", 2),
            data=data_subset,
            legend="full",
            alpha=0.3
        )
        plt.title('t-SNE Visualization', fontsize=14)
        plt.show()
    
    def run(self):
        self.explore_data()
        self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        model = RandomForestClassifier()  # Replace with your desired model
        model = self.train_model(model)
        self.evaluate_model(model)
        self.plot_confusion_matrix(model)
        self.plot_roc_curve(model)
        self.plot_precision_recall_curve(model)
        self.visualize_tsne()

if __name__ == '__main__':
    data_path ='C:/Users/ayala/Desktop/Front-Pfa1/Front-Pfa/treatement/creditcard.csv'  # Provide the path to your dataset
    fraud_detection = CreditCardFraudDetection(data_path)
    fraud_detection.run()

