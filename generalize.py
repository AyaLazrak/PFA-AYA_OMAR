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
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os

# Ignorer l'avertissement sur le nombre de cœurs
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

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
        sns.histplot(amount_val, ax=ax[0], color='r')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlim([min(amount_val), max(amount_val)])
        sns.histplot(time_val, ax=ax[1], color='b')
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

        # Afficher la distribution des classes avant SMOTE
        print("Distribution des classes avant SMOTE :")
        print(pd.Series(original_ytrain).value_counts())

        # Application de SMOTE pour équilibrer les classes dans l'ensemble d'entraînement
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(original_Xtrain, original_ytrain)
        self.X_test, self.y_test = original_Xtest, original_ytest

        # Afficher la distribution des classes après SMOTE
        print("Distribution des classes après SMOTE :")
        print(pd.Series(self.y_train).value_counts())

        return self.X_train, self.X_test, self.y_train, self.y_test
    
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

    def visualize_tsne(self, X, y, title):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(X)

        df_tsne = pd.DataFrame()
        df_tsne['tsne-2d-one'] = tsne_results[:, 0]
        df_tsne['tsne-2d-two'] = tsne_results[:, 1]
        df_tsne['Class'] = y

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="Class",
            palette=sns.color_palette("hsv", 2),
            data=df_tsne,
            legend="full",
            alpha=0.3
        )
        plt.title(title, fontsize=14)
        plt.show()
    
    def run(self):
        self.explore_data()
        self.preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data
        models = [
            RandomForestClassifier(),
            LogisticRegression(),
            # Ajoutez d'autres modèles ici si nécessaire
        ]

        for model in models:
            print(f"Training {model.__class__.__name__}")
            model = self.train_model(model)
            self.evaluate_model(model)
            self.plot_confusion_matrix(model)
            self.plot_roc_curve(model)
            self.plot_precision_recall_curve(model)
            self.visualize_learning_curve(model)
            self.visualize_tsne(self.X_train, self.y_train, 't-SNE Visualization')
            print("--------------------")


if __name__ == '__main__':
    data_path = 'C:/Users/ayala/Desktop/bank_data/treatement/creditcard.csv'  # Provide the path to your dataset
    fraud_detection = CreditCardFraudDetection(data_path)
    fraud_detection.run()
