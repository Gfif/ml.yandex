import pandas as pn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pn.read_csv('classification.csv')

TP = FN = FP = TN = 0
for row in df.iterrows():
	if row[1]['true'] == 1:
		if row[1]['pred'] == 1:
			TP += 1
		else:
			FN += 1
	else:
		if row[1]['pred'] == 1:
			FP += 1
		else:
			TN += 1

print TP, FP, FN, TN

print "%.2f %.2f %.2f %.2f" % (accuracy_score(df['true'], df['pred']), precision_score(df['true'], df['pred']), \
	recall_score(df['true'], df['pred']), f1_score(df['true'], df['pred']))