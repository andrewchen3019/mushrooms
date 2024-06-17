#Andrew, Junseo, William, Robert
#submitted by Junseo


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import cross_validate

#read file
df = pd.read_csv("mushrooms.csv")


#there are 2480 missing data points in stalk-root feature

print((df=='?').sum())

#plotting stalk-root feature. We see the missing ("?") data and we see the mode is "b". 

sns.countplot(data=df, x="stalk-root")
plt.show()

#replacing "?" with "b"
df['stalk-root']=df['stalk-root'].replace(['?'],['b'])


#plotting odor vs class, using hues for class. We show that p, f, c, y, s, and m are strongly correlated to poisonous mushrooms.The rest are strongly correlated to edible mushrooms. The key: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s

sns.countplot(x='odor', hue='class', data=df)
plt.show()

#plotting class. We show that there are more edible mushrooms. We need to balance (which we will do later in code)
sns.countplot(data=df, x="class")
plt.show()


#plotting class vs bruises using heatmap. We show that bruises are correlated to edible mushrooms and no bruises are correlated to poisonous mushrooms. "t" = bruise, "f" = no bruise. 

pivot_table = df.pivot_table(index='class', columns='bruises', aggfunc='size')
sns.heatmap(pivot_table, annot=True, fmt="d")
plt.show()


#converting all letters into numbers for every column

df['class'] = df['class'].replace(['e', 'p'], [0, 1])

df['cap-shape'] = df['cap-shape'].replace(['b', 'c', 'x', 'f', 'k', 's'], [0, 1, 2, 3, 4, 5])

df['cap-surface'] = df['cap-surface'].replace(['f', 'g', 'y', 's'], [0, 1, 2, 3])

df['cap-color'] = df['cap-color'].replace(['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

df['bruises'] = df['bruises'].replace(['t', 'f'], [0, 1])

df['odor'] = df['odor'].replace(['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'], [0, 1, 2, 3, 4, 5, 6, 7, 8])

df['gill-attachment'] = df['gill-attachment'].replace(['a', 'd', 'f', 'n'], [0, 1, 2, 3])

df['gill-spacing'] = df['gill-spacing'].replace(['c', 'w', 'd'], [0, 1, 2])

df['gill-size'] = df['gill-size'].replace(['b', 'n'], [0, 1])

df['gill-color'] = df['gill-color'].replace(['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

df['stalk-shape'] = df['stalk-shape'].replace(['e', 't'], [0, 1])

df['stalk-root'] = df['stalk-root'].replace(['b', 'c', 'u', 'e', 'z', 'r'], [0, 1, 2, 3, 4, 5])

df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].replace(['f', 'y', 'k', 's'], [0, 1, 2, 3])

df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].replace(['f', 'y', 'k', 's'], [0, 1, 2, 3])

df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], [0, 1, 2, 3, 4, 5, 6, 7, 8])

df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], [0, 1, 2, 3, 4, 5, 6, 7, 8])

df['veil-type'] = df['veil-type'].replace(['p', 'u'], [0, 1])

df['veil-color'] = df['veil-color'].replace(['n', 'o', 'w', 'y'], [0, 1, 2, 3])

df['ring-number'] = df['ring-number'].replace(['n', 'o', 't'], [0, 1, 2])

df['ring-type'] = df['ring-type'].replace(['c', 'e', 'f', 'l', 'n', 'p' ,'s', 'z'], [0, 1,2, 3, 4, 5, 6, 7])

df['spore-print-color'] = df['spore-print-color'].replace(['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], [0, 1, 2, 3, 4,5, 6, 7, 8])

df['population'] = df['population'].replace(['a', 'c', 'n', 's', 'v', 'y'], [0,1, 2 , 3 ,4, 5])

df['habitat'] = df['habitat'].replace(['g', 'l', 'm', 'p', 'u', 'w', 'd'], [0, 1, 2 ,3 ,4, 5, 6])

#doing data balancing, by resampling

df_1 = df[(df["class"] == 1)]
df_0 = df[(df["class"] == 0)]

print(df['class'].value_counts()[1]) #3916 class 1 (poison)
print(df['class'].value_counts()[0]) #4208 class 0 (edible)


df_1 = resample(df_1, n_samples= 4208, random_state=42)

df = pd.concat([df_1, df_0])

X = df.drop("class", axis=1)

y = df["class"]

#Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#change model
clf = tree.DecisionTreeClassifier() #GaussianNB() #LogisticRegression()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

# Plot Decision Tree (Deicison Tree only)
plt.figure(figsize=(8, 8)) 
tree.plot_tree(clf, fontsize=10) 
tree.plot_tree(clf)

plt.show()


#confusion matrix
cfm = confusion_matrix(pred, y_test)
print(cfm)

#get AUROC values
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred) 

roc_auc = metrics.auc(fpr, tpr) 

#plot AUROC using values
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='GNB')
display.plot()
plt.show()

#do crass validation
scoring=['f1','accuracy','precision','recall']
k=10
cv_results = cross_validate(estimator=clf, X=X, y=y, cv=k, scoring=scoring)

#print out results
print("f1:",cv_results["test_f1"].mean())
print("accuracy:", cv_results["test_accuracy"].mean())
print("precision:", cv_results["test_precision"].mean())
print("recall:", cv_results["test_recall"].mean())