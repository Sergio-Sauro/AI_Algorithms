import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from PIL import Image
import joblib

def representation():
    graph=graph_from_dot_data(dot_data)
    graph.write_png('tree.png')
    img=Image.open('/home/user/.../.../tree.png')                                                   #choose a path where you want to open the decission tree photo
    img.show()
    sb.barplot(columns,importances)
    plt.title('Magnitude weights')
    plt.show()

def guardar_modelo():
    joblib.dump(tree,'Decission_tree_trained.pkl')                                                  #Dump of your decission tree trained with the input data

if __name__ == "__main__":


    data=pd.read_csv('/home/user/.../prueba.data', usecols=['Variable1','Variable2','...','VariableN'])     #usecols=Names of the variables you are gonna use, including inputs and target.
    print(data.head(0))
    tree=DecisionTreeClassifier()
    X=data[['Variable1','Variable2','...','VariableN']]                                                 #All the INPUT columns
    Y=data[['VariableTarget']]                                                                          # The TARGET column

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=0)                  #Proportion 80% trainning - 20% test
    tree.fit(X_train,Y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    print('Precisión sobre conjunto de Train:', accuracy_score(y_train_pred,Y_train)*100,"%")
    print('Precisión sobre conjunto de Test: %.2f' % (accuracy_score(y_test_pred,Y_test)*100),"%")
    importances=tree.feature_importances_
    columns=X.columns

    feature_data=columns
    dot_data=export_graphviz(tree,  feature_names=feature_data, class_names=['b1_Ok','b1_REVISAR','b2_OK','b2_REVISAR'],filled=bool)

#These functions save the model in a .pkl file, show the decision tree and magnitude weights
#    guardar_modelo()
#    representation()
