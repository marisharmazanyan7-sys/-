from sklearn.tree import DecisionTreeClassifier
x_train = [[1],[2],[3],[4],[5],[6],[7]]
y_train = [0,0,0,1,1,1,0]
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
x_new = [[3], [5.5], [7]]
y_new = [0,1,0]
predictions = clf.predict(x_new)
print("new data:", x_new)
print("predictions:", predictions)
print("actual labels:", y_new)
print("accuracy on new data:", clf.score(x_new,y_new))

from sklearn.linear_model import LinearRegression
x = [[1],[2],[3],[4],[5]]
y = [40,50,60,70,80]
model = LinearRegression()
model.fit(x,y)
predictions = model.predict([[6.5]])
print("Predicted score for 6.5 hours studied:", predictions[0])
print("R^2 score:", model.score(x,y))
import matplotlib.pyplot as plt 
plt.scatter(x,y, color="blue",label="data points")
plt.plot(x, model.predict(x),color= "red", label="regretion line")
plt.scatter(6,predictions,color="green",s=100,marker="*",label="prediction (6h)")
plt.xlabel("hours studied")
plt.ylabel("exam score")
plt.show()