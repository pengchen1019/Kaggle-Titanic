# 导入库
import pandas as pd

import re

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# 导入数据
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# 数据数量
train_df.info()

# 数据总体信息
data = pd.DataFrame(train_df.describe())
data.to_csv("data1.csv")

# 样本前十位数据
data = pd.DataFrame(train_df.head(10))
data.to_csv("data2.csv")

# 缺失数据占比
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum() / train_df.isnull().count() * 100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
data = pd.DataFrame(missing_data.head(5))
data.to_csv("data3.csv")

# 查看数据类别名称
print(train_df.columns.values)
print("-" * 100)

# 年龄/性别与存活率的关系
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
women = train_df[train_df['Sex'] == 'female']
men = train_df[train_df['Sex'] == 'male']
ax = sns.histplot(women[women['Survived'] == 1].Age.dropna(),
                  bins=18, label=survived, ax=axes[0], kde=False)
ax = sns.histplot(women[women['Survived'] == 0].Age.dropna(),
                  bins=40, label=not_survived, ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')
ax = sns.histplot(men[men['Survived'] == 1].Age.dropna(),
                  bins=18, label=survived, ax=axes[1], kde=False)
ax = sns.histplot(men[men['Survived'] == 0].Age.dropna(),
                  bins=40, label=not_survived, ax=axes[1], kde=False)
ax.legend()
_ = ax.set_title('Male')
# plt.show()

# 登船地点/舱层/性别与存活率的关系
FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None, hue_order=None)
FacetGrid.add_legend()
# plt.show()

# 舱层与存活率的关系
sns.barplot(x='Pclass', y='Survived', data=train_df)
# plt.show()

# 进一步探究舱层/年龄与存活率的关系
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# plt.show()

# Sibsp and parch
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

print(train_df['not_alone'].value_counts())
print("-" * 100)

axes = sns.catplot('relatives', 'Survived', data=train_df, aspect=2.5, )
# plt.show()

train_df = train_df.drop(['PassengerId'], axis=1)
# 填补cabin空缺
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
# 填补age空缺
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

print(train_df["Age"].isnull().sum())
print(train_df['Embarked'].describe())
print("-" * 100)

common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

# train_df.info()
data = [train_df, test_df]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

genders = {"male": 0, "female": 1}
data = [train_df, test_df]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# 为年龄分组
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[dataset['Age'] > 66, 'Age'] = 6

print(train_df['Age'].value_counts())
print("-" * 100)

data = pd.DataFrame(train_df.head(10))
data.to_csv("data5.csv")

data = [train_df, test_df]
for dataset in data:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
    dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

print(train_df['Fare'].value_counts())
print("-" * 100)

data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class'] = dataset['Age'] * dataset['Pclass']

for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare'] / (dataset['relatives'] + 1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

print(train_df.head(10))
print("-" * 100)

# 建立机器学习模型
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# 随机梯度下降
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred1 = sgd.predict(X_test)

sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# 随机森林
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# 逻辑回归分析
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# k-近邻算法
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

Y_pred2 = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# 朴素贝叶斯分类器
Y_pred3 = gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# 感知机
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)

Y_pred4 = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# 支持向量机
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred5 = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# 决策树
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred6 = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Decison Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, acc_random_forest,
              acc_gaussian, acc_perceptron, acc_sgd, acc_decision_tree]
})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))
print("-" * 100)

# 10折交叉验证
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring="accuracy")
print("scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("-" * 100)

importance = pd.DataFrame({
    'feature': X_train.columns, 'importance': np.round(random_forest.feature_importances_, 3)
})
importance = importance.sort_values('importance', ascending=False).set_index('feature')
print(importance.head(15))
print("-" * 100)
importance.plot.bar()
# plt.show()

train_df = train_df.drop("not_alone", axis=1)
test_df = test_df.drop("not_alone", axis=1)
train_df = train_df.drop("Parch", axis=1)
test_df = test_df.drop("Parch", axis=1)

# 随机森林再训练
random_forest = RandomForestClassifier(n_estimators=100, oob_score=True)
random_forest.fit(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest, 2), "%")
print("-" * 100)

print("oob score:", round(random_forest.oob_score_, 4) * 100, "%")
print("-" * 100)

# 超参数调整
# X_train = train_df.drop("Survived", axis=1)
# Y_train = train_df["Survived"]
# param_grid = {
#     "criterion": ["gini", "entropy"],
#     "min_samples_leaf": [1, 5, 10, 25, 50, 70],
#     "min_samples_split": [2, 4, 10, 12, 16, 18, 25, 35],
#     "n_estimators": [100, 400, 700, 1000, 1500]
# }
# from sklearn.model_selection import GridSearchCV, cross_val_score
#
# rf = RandomForestClassifier(n_estimators=100, max_features='auto',
#                             oob_score=True, random_state=1, n_jobs=-1)
# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
# clf.fit(X_train, Y_train)
# print(clf.best_params_)

# 测试新参数
random_forest = RandomForestClassifier(criterion="gini",
                                       min_samples_leaf=5,
                                       min_samples_split=12,
                                       n_estimators=400,
                                       max_features='auto',
                                       oob_score=True,
                                       random_state=1,
                                       n_jobs=-1
                                       )
random_forest.fit(X_train, Y_train)
Y_prediction1 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
print("oob score:", round(random_forest.oob_score_, 4) * 100, "%")
print("-" * 100)

# 混淆矩阵
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
print(confusion_matrix(Y_train, predictions))

# 准确率和召回率
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:", recall_score(Y_train, predictions))
print("-" * 100)

# F-score
from sklearn.metrics import f1_score

print(f1_score(Y_train, predictions))
print("-" * 100)

# PR曲线
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:, 1]
precision, recall, threshold = precision_recall_curve(Y_train, y_scores)


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])


plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

# ROC 和 AUC 曲线
from sklearn.metrics import roc_curve

false_postive_rate, true_postive_rate, threshold = roc_curve(Y_train, y_scores)


def plot_roc_curve(false_postive_rate, true_positive_rate, label=None):
    plt.plot(false_postive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)


plt.figure(figsize=(14, 7))
plot_roc_curve(false_postive_rate, true_postive_rate)
# plt.show()

# ROC 和 AUC 值
from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
print("-" * 100)

# 使用模型对数据进行预测
y_test = random_forest.predict(X_test)

# 乘客id
passenger_id = test_df['PassengerId']
# 预测结果转换为int类型
y_test = y_test.astype(int)

# 将结果做为新的数据保存
pred_df = pd.DataFrame(
    {'PassengerId': passenger_id,
     'Survived': y_test})
pred_df.shape

# 保存到本地
pred_df.to_csv("prediction.csv", index=False)
