from sklearn.metrics import accuracy_score,f1_score,classification_report
def classification_metrics(y_true,y_pred): return {"accuracy":float(accuracy_score(y_true,y_pred)),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted")),"macro_f1":float(f1_score(y_true,y_pred,average="macro"))}
def report(y_true,y_pred,labels): return classification_report(y_true,y_pred,labels=list(range(len(labels))),target_names=list(labels),zero_division=0)
