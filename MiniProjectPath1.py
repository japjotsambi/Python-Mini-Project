import numpy as np
# from cluster import createClusters
# from point import makePointList
# from kmeans import kmeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
#print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data

# QUESTION 1 : GROUPING STUDENTS BY THEIR VIDEO-WATCHING BEHAVIOR

# use students that complete at least five of the videos in analysis

filtered_data = dataset_1.groupby('userID').filter(lambda x: len(x) >= 5)

# Cluster students
behaviors = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs','numFFs']
# means
student_behaviors = filtered_data.groupby('userID')[behaviors].mean()
student_data = student_behaviors.to_numpy()

# KMEANS

# Gaussian

def gaus_mixture(data, n_components_list):
    best_bic = np.inf
    best_k = None
    bics = []

    for k in n_components_list:
        gm = GaussianMixture(n_components=k, random_state=0).fit(data)
        bic = gm.bic(data)
        bics.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return best_k, bics

# Try different numbers of clusters
n_components_list = [2, 3, 4, 5, 6, 7, 8]

# Find best number of clusters by BIC
best_k, bics = gaus_mixture(student_data, n_components_list)
print(f"\n[Gaussian Mixture Model] Best number of clusters (by BIC): {best_k}")

# Plot BIC curve
plt.figure()
plt.plot(n_components_list, bics, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('BIC')
plt.title('BIC vs Number of Clusters for GMM')
plt.grid(True)
plt.show()

# Train final GMM
gmm = GaussianMixture(n_components=best_k, random_state=0)
gmm.fit(student_data)
cluster_labels = gmm.predict(student_data)

# Analyze clusters
for i in range(best_k):
    cluster_mean = np.mean(student_data[cluster_labels == i], axis=0)
    print(f"Cluster {i}: {np.sum(cluster_labels == i)} students")
    print(f"  Average behavior: {np.round(cluster_mean, 4)}")

# Functions you already had (fixed for this project)
def conf_matrix(y_pred, y_true, num_class):
    y_pred = np.array(y_pred, dtype=int)
    y_true = np.array(y_true, dtype=int)
    matrix = np.zeros((num_class, num_class), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

def get_model(name, params):
    if name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params)
    elif name == "SVM":
        rand_state, prob = params
        model = SVC(random_state=rand_state, probability=prob)
    elif name == "MLP":
        hl_sizes, rand_state, act_func = params
        model = MLPClassifier(hidden_layer_sizes=hl_sizes, random_state=rand_state, activation=act_func)
    else:
        model = None
    return model

def get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class=2):
    model = get_model(model_name, params)
    model.fit(train_data, train_labels)
    y_pred = model.predict(test_data)
    acc = metrics.accuracy_score(test_labels, y_pred)
    conf_mat = conf_matrix(y_pred, test_labels, num_class)
    probabilities = model.predict_proba(test_data)[:,1]
    auc_scor = metrics.roc_auc_score(test_labels, probabilities)
    return acc, conf_mat, auc_scor

# ============================
# Load and prepare the data
# ============================

# Load behavior-performance.txt
with open('behavior-performance.txt', 'r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pd.DataFrame.from_records(raw_data[1:], columns=raw_data[0])

# Convert numeric columns properly
df['VidID'] = pd.to_numeric(df['VidID'])
df['fracSpent'] = pd.to_numeric(df['fracSpent'])
df['fracComp'] = pd.to_numeric(df['fracComp'])
df['fracPaused'] = pd.to_numeric(df['fracPaused'])
df['numPauses'] = pd.to_numeric(df['numPauses'])
df['avgPBR'] = pd.to_numeric(df['avgPBR'])
df['numRWs'] = pd.to_numeric(df['numRWs'])
df['numFFs'] = pd.to_numeric(df['numFFs'])
df['s'] = pd.to_numeric(df['s'])

# Define features
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']

# ============================
# Part 1: Predicting average score per student
# ============================

# Keep only students who completed at least 5 videos
filtered = df.groupby('userID').filter(lambda x: len(x) >= 5)

# Group by student and calculate mean behavior and mean score
student_data = filtered.groupby('userID')[features + ['s']].mean()

X1 = student_data[features].values
y1 = (student_data['s'] >= 0.5).astype(int).values  # 1 if >= 50% correct, else 0

# Split into training and testing
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)

# Train and evaluate models for Part 1
print("Part 1: Predicting average student score")
for model_name, params in [("KNN", 3), ("SVM", [0, True]), ("MLP", [(15,10), 1, "relu"])]:
    acc, conf, auc = get_model_results(model_name, params, X1_train, y1_train, X1_test, y1_test)
    print(f"{model_name}: Accuracy={acc:.4f}, AUROC={auc:.4f}")
    print(conf)

# ============================
# Part 2: Predicting quiz question correctness
# ============================

# Use all student-video pairs for per-quiz prediction
X2 = df[features].values
y2 = df['s'].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

# Train and evaluate models for Part 2
print("\nPart 2: Predicting per-quiz correctness")
for model_name, params in [("KNN", 3), ("SVM", [0, True]), ("MLP", [(15,10), 1, "relu"])]:
    acc, conf, auc = get_model_results(model_name, params, X2_train, y2_train, X2_test, y2_test)
    print(f"{model_name}: Accuracy={acc:.4f}, AUROC={auc:.4f}")
    print(conf)


'''
QUESTIONS TO BE ANSWERED
1. How well can the students be naturally grouped or clustered by their video-watching behavior (fracSpent, fracComp, fracPaused,
numPauses, avgPBR, numRWs, and numFFs)? You should use all students that complete at least five of the videos in your analysis.
Hints: KMeans or distribution parameters(mean and standard deviation) of Gaussians

2. Can student's video-watching behavior be used to predict a student's performance (i.e., average score s across all quizzes)?
Will adding the cluster information (e.g. which group they belong) help the model improve the performance and makes the model become
more/less under/overfit? Explain your conclusion and discuss the reasons why such results could happen.

3. Taking this a step further, how well can you predict a student's performance on a particular in-video quiz question
(i.e., whether they will be correct or incorrect) based on their video-watching behaviors while watching the corresponding video?
You should use all student-video pairs in your analysis.
'''

'''
This	file	specifies	the	meaning	of	the	data	fields	in	behavior-performance.txt.	
Each	row	of	behavior-performance.txt	corresponds	to	one	student-video	pair.	There	are	10	fields:	
userID,	VidID,	fracSpent,	fracComp,	fracPaused,	numPauses,	avgPBR,	numRWs,	
numFFs,	and	s.	These	are	specifically	defined	and	calculated	as	follows:	
• userID	is	the	(anonymized)	ID	of	the	student.	Each	user	appears	many	times	in	the	dataset.
 • videoID	is	the	ID	of	the	video,	between	0	and	92.	In	other	words,	there	are	93	videos.
 • fracSpent	is	the	fraction	of	time	the	student	spent	watching	the	video	(either	playing	or
 paused),	relative	to	the	length	of	the	video.	So,	if	someone	was	on	a	3	minute	video	for	4
 minutes,	this	would	be	fracSpent=4/3.
 • fracComp	is	the	fraction	of	the	video	the	student	watched,	between	fracComp=0	(none)
 and	fracComp>=0.9	(completely).
 • fracPaused	is	the	fraction	of	time	the	student	spent	paused	on	the	video,	relative	to	the
 length	of	the	video.	So	if	someone	was	paused	for	0.5	min	on	a	3	minute	video,	this	would	be
 fracPaused=1/6.
 • numPauses	is	the	number	of	times	the	student	paused	the	video.
 • avgPBR	is	the	average	playback	rate	that	the	student	used	while	watching	the	video,	ranging
 between	0.5x	and	2.0x.
 • numRWs	is	the	number	of	times	the	student	skipped	backwards	(rewind)	in	the	video.
 • numFFs	is	the	number	of	times	the	student	skipped	forward	(fast	forward)	in	the	video.
 • s	is	whether	the	student	was	correct	(s=1)	or	not	(s=0)	on	their	first	attempt	at	answering
 the	question	given	directly	after	the	video.
 • fracPlayed	is	the	fraction of the video watched by the student
'''