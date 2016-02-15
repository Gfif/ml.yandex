import pandas as pn
from sklearn.metrics import roc_auc_score, precision_recall_curve

df = pn.read_csv('scores.csv')

algs = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']

preds = [df['score_logreg'], df['score_svm'], df['score_knn'], df['score_tree']]
auc_scores = map(lambda x: "%.2f" % roc_auc_score(df['true'], x), [df[a] for a in algs])

index_of_max = max([(i, score) for i, score in enumerate(auc_scores)], key=lambda x: x[1])[0]
print algs[index_of_max]

curves = map(lambda x: precision_recall_curve(df['true'], x), preds)

results = []
for curve in curves:
	precision, recall, thresholds = curve
	precision_recall = zip(precision, recall)
	results.append(max(precision_recall, key=lambda x: x[0] if x[1] >= 0.7 else 0)[0])

index_of_max = max([(i, score) for i, score in enumerate(results)], key=lambda x: x[1])[0]
print algs[index_of_max]