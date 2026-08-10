import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('titanic.csv')


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna('S')
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)


plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=list(X.columns), class_names=[
          'Died', 'Survived'], filled=True, rounded=True, fontsize=10)
plt.title("Titanic Survival Decision Tree Logic")
plt.savefig('decision_tree.png', bbox_inches='tight', dpi=200)
print("Successfully saved decision_tree.png to your folder.")


plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=[
                                      'Died', 'Survived'], cmap='Blues')
plt.title("Confusion Matrix (Test Data Performance)")
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=200)
print("Successfully saved confusion_matrix.png to your folder.")
