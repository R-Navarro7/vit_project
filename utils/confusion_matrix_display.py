import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, acc, target_names, file_name):

	annot = True if len(cm[0]) < 10 else False
	# plot the confusion matrix using seaborn
	sns.set(rc={'figure.figsize': (10, 6)})  # Size in inches
	sns.heatmap(cm, annot=annot, cmap='Blues', fmt='.2f')
	if not annot:
		plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title(f'Confusion Matrix. Accuracy: {acc}')
	plt.savefig('./Results/CMs/' + file_name + '.png')
