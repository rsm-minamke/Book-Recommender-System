# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


allRatings = []
for l in readCSV(r"C:\Users\mrun7\Downloads\assignment1 (1)\assignment1\train_Interactions.csv.gz"):
    allRatings.append(l)

len(allRatings)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# +
##################################################
# Read prediction                                #
##################################################

# +
# From baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV(r"C:\Users\mrun7\Downloads\assignment1 (1)\assignment1\train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

# +
# Generate a negative set

userSet = set()
bookSet = set()
readSet = set()

for u,b,r in allRatings:
    userSet.add(u)
    bookSet.add(b)
    readSet.add((u,b))

lUserSet = list(userSet)
lBookSet = list(bookSet)

notRead = set()
for u,b,r in ratingsValid:
    #u = random.choice(lUserSet)
    b = random.choice(lBookSet)
    while ((u,b) in readSet or (u,b) in notRead):
        b = random.choice(lBookSet)
    notRead.add((u,b))

readValid = set()
for u,b,r in ratingsValid:
    readValid.add((u,b))

# +
from sklearn.metrics.pairwise import cosine_similarity

def compute_features_enhanced(user, book, ratingsPerUser, ratingsPerItem, bookCount, bookSet):
    # User Features
    user_activity = len(ratingsPerUser[user])
    avg_user_similarity = np.mean([
        Jaccard(set(ratingsPerItem[b1]), set(ratingsPerItem[b2]))
        for b1, _ in ratingsPerUser[user]
        for b2, _ in ratingsPerUser[user] if b1 != b2
    ]) if len(ratingsPerUser[user]) > 1 else 0

    # Book Features
    popularity_score = bookCount[book] / max(bookCount.values())
    book_rank = list(bookCount.keys()).index(book) / len(bookCount)

    # Interaction Features
    interaction_score = user_activity * popularity_score
    max_jaccard = max([
        Jaccard(set(ratingsPerItem[book]), set(ratingsPerItem[b]))
        for b, _ in ratingsPerUser[user]
    ]) if user_activity > 0 else 0
    avg_jaccard = np.mean([
        Jaccard(set(ratingsPerItem[book]), set(ratingsPerItem[b]))
        for b, _ in ratingsPerUser[user]
    ]) if user_activity > 0 else 0
    max_cosine = max([
        cosine_similarity(
            np.array([1 if b == book else 0 for b in bookSet]).reshape(1, -1),
            np.array([1 if b == other else 0 for b in bookSet]).reshape(1, -1)
        )[0][0] for other, _ in ratingsPerUser[user]
    ]) if user_activity > 0 else 0
    avg_cosine = np.mean([
        cosine_similarity(
            np.array([1 if b == book else 0 for b in bookSet]).reshape(1, -1),
            np.array([1 if b == other else 0 for b in bookSet]).reshape(1, -1)
        )[0][0] for other, _ in ratingsPerUser[user]
    ]) if user_activity > 0 else 0

    # Combine Features
    return [
        user_activity, avg_user_similarity, popularity_score, book_rank,
        interaction_score, max_jaccard, avg_jaccard, max_cosine, avg_cosine
    ]


# +
feature_matrix = []
labels = []

for (label, sample) in [(1, readValid), (0, notRead)]:
    for (user, book) in sample:
        features = compute_features_enhanced(user, book, ratingsPerUser, ratingsPerItem, bookCount, bookSet)
        feature_matrix.append(features)
        labels.append(label)

feature_matrix = np.array(feature_matrix)
labels = np.array(labels)

# +
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Prepare feature matrix and labels
feature_matrix = np.array(feature_matrix)
labels = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)

# Train XGBoost model
xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# Validate the model
y_pred = xgb.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy with Enhanced Features:", val_accuracy)

# +
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Initialize XGBoost model
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1  # Use all available cores
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Print best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# +
predictions = open(r"C:\Users\mrun7\Downloads\assignment1 (1)\assignment1\predictions_Read.csv", 'w')
for l in open(r"C:\Users\mrun7\Downloads\assignment1 (1)\assignment1\pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # Compute features for the user-book pair
    # Compute features for the user-book pair
    features = compute_features_enhanced(u, b, ratingsPerUser, ratingsPerItem, bookCount, bookSet)

    # Ensure the feature array has the correct shape (reshape for prediction)
    features = np.array(features).reshape(1, -1)

    # Predict using the best model from GridSearchCV
    pred = grid_search.best_estimator_.predict(features)[0]

    # Write the prediction to the file
    predictions.write(u + ',' + b + ',' + str(pred) + '\n')

# Close the predictions file
predictions.close()

# +
##################################################
# Rating prediction                              #
##################################################

# +
trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)

validMSE = 0
for u,b,r in ratingsValid:
  se = (r - globalAverage)**2
  validMSE += se

validMSE /= len(ratingsValid)

betaU = {}
betaI = {}
for u in ratingsPerUser:
    betaU[u] = 0

for b in ratingsPerItem:
    betaI[b] = 0
    
alpha = globalAverage # Could initialize anywhere, this is a guess
def iterate(lamb):
    newAlpha = 0
    for u,b,r in ratingsTrain:
        newAlpha += r - (betaU[u] + betaI[b])
    alpha = newAlpha / len(ratingsTrain)
    for u in ratingsPerUser:
        newBetaU = 0
        for b,r in ratingsPerUser[u]:
            newBetaU += r - (alpha + betaI[b])
        betaU[u] = newBetaU / (lamb + len(ratingsPerUser[u]))
    for b in ratingsPerItem:
        newBetaI = 0
        for u,r in ratingsPerItem[b]:
            newBetaI += r - (alpha + betaU[u])
        betaI[b] = newBetaI / (lamb + len(ratingsPerItem[b]))
    mse = 0
    for u,b,r in ratingsTrain:
        prediction = alpha + betaU[u] + betaI[b]
        mse += (r - prediction)**2
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for b in betaI:
        regularizer += betaI[b]**2
    mse /= len(ratingsTrain)
    return mse, mse + lamb*regularizer

mse,objective = iterate(1)
newMSE,newObjective = iterate(1)
iterations = 2
while iterations < 10 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(1)
    iterations += 1

validMSE = 0
for u,b,r in ratingsValid:
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    prediction = alpha + bu + bi
    validMSE += (r - prediction)**2

validMSE /= len(ratingsValid)
print("Validation MSE = " + str(validMSE))

iterations = 1
while iterations < 10 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(5)
    iterations += 1
    
validMSE = 0
for u,b,r in ratingsValid:
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    prediction = alpha + bu + bi
    validMSE += (r - prediction)**2

validMSE /= len(ratingsValid)
print("Validation MSE = " + str(validMSE))

# +
predictions = open(r"C:\Users\mrun7\Downloads\assignment1 (1)\assignment1\predictions_Rating.csv", 'w')
for l in open(r"C:\Users\mrun7\Downloads\assignment1 (1)\assignment1\pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # Retrieve user and item biases, or default to 0 if not found
    u,b = l.strip().split(',')
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

predictions.close()
# -

pip install xgboost
