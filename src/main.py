import pandas as pd
import string
import torch
import torch.nn as nn
import numpy as np
import random


from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


#1. Load processed CSV
df = pd.read_csv("../data/processed/spam.csv")
print(df.head())

#2. clean (process) the text 
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

#3. TF-IDF vectorization 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values
print("Feature shape:" , X.shape)

#4. train-test split 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

#5.convert to tensor 
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#6. build model 
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Model(X_train.shape[1])

#7. loss+optimizer 
pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#8. train model 
epochs = 20

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#9. evaluate model 
with torch.no_grad():
    predictions = torch.sigmoid(model(X_test))
    predicted = (predictions > 0.5).float()

    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print("Accuracy:", accuracy)

#10. confusion matrix 
y_pred = predicted.numpy()
y_true = y_test.numpy()

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

#11.save model 
torch.save(model.state_dict(), "../models/model.pt")

print("Model saved successfully ✅")

#12. test with custom inputs 

def predict_spam(text):
    # same preprocessing
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # convert to TF-IDF using SAME vectorizer
    vector = vectorizer.transform([text]).toarray()
    vector = torch.tensor(vector, dtype=torch.float32)

    # prediction
    with torch.no_grad():
        output = torch.sigmoid(model(vector))
        prediction = (output > 0.5).float().item()

    if prediction == 1:
        print("Spam ❌")
    else:
        print("Not Spam ✅")


# text samples 
predict_spam("Congratulations! You won a free iPhone")
predict_spam("Hey, are we meeting tomorrow?")
predict_spam("Get rich quick offer limited time")
predict_spam("Ammi ne khana bana liya hai")
predict_spam("Call me when you reach home")