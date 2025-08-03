import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc,mean_squared_error
from sklearn.ensemble import RandomForestClassifier

import pickle
# Load your data
file_path = "D:\\projects\\Predict the apple\\projects ml\\apple_quality.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=['resa'])

X = df[['size', 'weight', 'sweet', 'crunch', 'juicines', 'rip', 'acidity']]
y = df['resa']


plt.figure(figsize=(7,5))
sns.countplot(data=df,x='resa')
plt.title('Equilibrium')
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

train_size = len(X_train)
test_size = len(X_test)

# Create a pie chart
labels = [f'Training Set\n({train_size} points)', f'Testing Set\n({test_size} points)']
sizes = [train_size, test_size]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Distribution of Data: Training Set vs Testing Set\n\n')
plt.show()


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)
# i use neural network i have 3 hiden layer
# and i have 7 node in input layer
# i have 1 output layer
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=7))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) #output of layer just 1 or 0
sgd = SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=321, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Predictions and thresholding
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
threshold = 0.5
y_pred_test_binary = np.where(y_pred_test > threshold, 1, 0)
y_pred_train_binary = np.where(y_pred_train > threshold, 1, 0)

# Confusion matrices and visualization for testing set
conf_matrix_test = confusion_matrix(y_test, y_pred_test_binary)
plt.figure(figsize=(9, 7))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix - Testing")
plt.show()

# Check for NaN values in y_train and y_pred_train_binary
print("NaN values in y_train:", np.isnan(y_train).any())
print("NaN values in y_pred_train_binary:", np.isnan(y_pred_train_binary).any())

# Confusion matrix and visualization for training set
conf_matrix_train = confusion_matrix(y_train, y_pred_train_binary)
plt.figure(figsize=(9, 7))
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix - Training")
plt.show()

# Classification report for testing set
print("Classification Report - Testing:")
print(classification_report(y_test, y_pred_test_binary))

precision = conf_matrix_test[1, 1] / (conf_matrix_test[1, 1] + conf_matrix_test[0, 1])
recall = conf_matrix_test[1, 1] / (conf_matrix_test[1, 1] + conf_matrix_test[1, 0])
sensitivity = recall  # Sensitivity is the same as recall
specificity = conf_matrix_test[0, 0] / (conf_matrix_test[0, 0] + conf_matrix_test[0, 1])
negative_predictive_value = conf_matrix_test[0, 0] / (conf_matrix_test[0, 0] + conf_matrix_test[1, 0])

print("Predictive:", negative_predictive_value)
print("Precision:", precision)
print("Sensitivity:", recall)
print("Specificity:", specificity)



#roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_binary)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - testing')
plt.legend(loc="lower right")
plt.show()

#trianing loss curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Trianing loss')
plt.plot(history.history['val_loss'],label='testing loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()


final_loss = model.evaluate(X_test, y_test)

print(f'Final Loss on Test Data: {final_loss}')


final_training_accuracy = history.history['accuracy'][-1]

print(f'Final Training Accuracy: {final_training_accuracy}')
y_pred_train = model.predict(X_train)
y_pred_train_binary = np.where(y_pred_train > threshold, 1, 0)

mse_train = mean_squared_error(y_train, y_pred_train_binary)

print(f'Mean Squared Error for Training: {mse_train}')

final_training_loss = history.history['loss'][-1]

print(f'Final Training Loss: {final_training_loss}')

accuracy_test = accuracy_score(y_test, y_pred_test_binary)
print(f'Accuracy for Testing: {accuracy_test:.4f}')

mse_test = mean_squared_error(y_test, y_pred_test_binary)
print(f'Mean Squared Error for Testing: {mse_test}')

for epoch in range(0, len(history.history['accuracy']), 10):
    print(f"Epoch {epoch + 1} - Training Loss: {history.history['loss'][epoch]:.4f}, Training Accuracy: {history.history['accuracy'][epoch]:.4f}, Validation Loss: {history.history['val_loss'][epoch]:.4f}, Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

categories = ['Training MSE', 'Testing MSE', 'Training Accuracy', 'Testing Accuracy']
values = [mse_train, mse_test, final_training_accuracy, accuracy_test]

# Plotting the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])

# Adding values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

# Adding labels and title
plt.ylabel('Values')
plt.title('MSE and Accuracy for Training and Testing Data')
plt.show()

user_data = pd.DataFrame(columns=['size', 'weight', 'sweet', 'crunch', 'juicines', 'rip', 'acidity'])

# Get user input for each feature
for feature in user_data.columns:
   user_input = float(input(f"Enter value for {feature}: "))
   user_data[feature] = [user_input]

user_data_scaled = scaler.transform(user_data)

user_prediction = model.predict(user_data_scaled)
user_prediction_binary = np.where(user_prediction > threshold, 1, 0)

#      Display the prediction for the user input
if user_prediction_binary[0] == 1:
    print("Predicted class for the input data: Good")
else:
    print("Predicted class for the input data: Bad")




pickle.dump(classifier, open("mltrainig.pkl", "wb"))