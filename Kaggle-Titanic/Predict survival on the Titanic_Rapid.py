import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# 分别读取训练数据和测试数据
train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')
# 测试读取是否成功
# print(train.info())
# print(test.info())

# 人工选取对预测有效的特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']

# 观察得知Embarked特征值缺失，需要补充
# print(X_train['Embarked'].value_counts())
# print(X_test['Embarked'].value_counts())

# 对于Embarked这种类别型的特征，使用出现频率最高的特征值来填充，这是可以相对减少引入误差的方法
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

# 对于Age这种类别型的特征，使用中位数或者平均数来填充缺失值，这也是可以相对减少引入误差的方法
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# 重新检查训练数据和测试数据，一切就绪
# print(X_train.info())
# print(X_test.info())

# 采用DictVectorizer对特征向量化
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

rfc = RandomForestClassifier()
xgbc = XGBClassifier()

# 使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier和XGBClassifier进行性能评估，并获取平均分类准确性的评分
cross_val_score(rfc, X_train, y_train, cv=5).mean()
cross_val_score(xgbc, X_train, y_train, cv=5).mean()

# 使用默认的RandomForestClassifier进行预测
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
# 将默认配置的RandomForestClassifier对测试数据的预测结果存储在文件rfc_submission.csv中
rfc_submission.to_csv('Dataset/rfc_submission.csv', index=False)

# 使用默认的XGBClassifier进行预测
xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
# 将默认配置的XGBClassifier对测试数据的预测结果存储在文件xgbc_submission.csv中
xgbc_submission.to_csv('Dataset/xgbc_submission.csv', index=False)

# 使用并行网络搜索的方式寻找更好的超参数组合，以期待进一步提高XGBClassifier的预测性能
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train, y_train)

# 查询优化后的XGBClassifier超参数配置以及交叉验证的准确性
# print(gs.best_score_)
# print(gs.best_params_)

# 使用经过优化超参数配置的XGBClassifier对测试数据的预测结果存储在xgbc_best_submission.csv中
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('Dataset/xgbc_best_submission.csv', index=False)