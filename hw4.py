import	csv
import	sys
import	warnings
import	numpy		as	np
import	threading	as	th

warnings.filterwarnings('error')

header = np.array(list(csv.reader(open('header.csv', 'rt'))))
train = np.array(list(csv.reader(open('train.csv', 'rt'))))
test = np.array(list(csv.reader(open('test.csv', 'rt'))))

# build training set
def buildX(train, features) :
	result = []
	for t in train :
		data = []
		for i in feature_list:
			data.append(float(t[i]))

		result.append(np.array(data))

	return np.array(result)

# build list of genres
def buildGenres(y):
	genres = np.unique(y)
	y_index = []
	for genre in y:
		y_index.append(int(np.searchsorted(genres, genre)))

	return np.array(genres), np.array(y_index)

# get Gaussian Naive Bayesian classifier
def getGaussianNB(X, y, start, end):
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	clf.fit(X[start:end], y[start:end])
	return clf

# get k-Nearest Neighbors classifier
def getNeighbors(X, y, start, end):
	from sklearn import neighbors
	n_neighbors = 15
	clf = neighbors.KNeighborsClassifier(n_neighbors, )
	clf.fit(X, y)
	return clf

# get Gaussian Mixture Model classifier
def getGMM(X, y, start, end):
	from sklearn.mixture import GMM
	clf = GMM(n_components=len(genres), covariance_type='tied', init_params='mc', n_iter=20)
	clf.fit(X[start:end], y[start:end])
	print "Gaussian Mixture Model self-test."
	y_pred = clf.predict(X[start:end])
	print "Accuracy:", np.mean(y_pred.ravel() == y_index[start:end].ravel()) * 100
	return clf

# get Hyperplanes (SVC) classifier (takes long time to fit)
def getHyperplanes(X, y, start, end):
	from sklearn.svm import SVC
	clf = SVC(kernel='linear')
	clf.fit(X[start:end], y[start:end])
	return clf

def getGMMPredictions(target, clf, start, end, feature_list):
	# this is to be used if our classifier is
	# built using Gaussian Mixture Model
	tested[target] = end-start
	for t in test[start:end]:
		test_data = []
		for i in feature_list:
			test_data.append(float(t[i]))

		predicted_genre = clf.predict(np.array(test_data).reshape(1, -1))
		if genres[predicted_genre][0] == t[0]:
			true_pos[np.searchsorted(genres, genres[predicted_genre][0])][target] += 1
		else :
			false_pos[np.searchsorted(genres, genres[predicted_genre][0])][target] += 1
			false_neg[np.searchsorted(genres, t[0])][target] += 1

def getPredictions(target, clf, start, end, feature_list):
	tested[target] = end-start
	for t in test[start:end]:
		test_data = []
		for i in feature_list:
			test_data.append(float(t[i]))

		predicted_genre = clf.predict(np.array(test_data).reshape(1, -1))[0]
		if predicted_genre == t[0]:
			true_pos[np.searchsorted(genres, predicted_genre)][target] += 1
		else :
			false_pos[np.searchsorted(genres, predicted_genre)][target] += 1
			false_neg[np.searchsorted(genres, t[0])][target] += 1

# select a lsit of features to use to build model
#feature_list = range(4, 34)
#feature_list = [6, 18, 20, 28]
feature_list = [5, 17, 19, 27]
print "Features used: ",
for f in feature_list :
	print header[0][f], ',', 
print ''

# build training set
X = buildX(train, feature_list)
y = train[:, 0] # Choose genre for output
genres, y_index = buildGenres(y)

# prepare arrays for results
true_pos = np.zeros((len(genres),4))
false_pos = np.zeros_like(true_pos)
false_neg = np.zeros_like(true_pos)

# pick a length of the train set
start = 0
end = len(X)-1

# pick a classifier
clf = getGaussianNB(X, y, start, end)
#clf = getNeighbors(X, y, start, end)
#clf = getGMM(X, y, start, end)
#clf = getHyperplanes(X, y, start, end) # clf.fit takes over six hours on full training set
print 'Built model from', (end-start), 'entries. Testing', len(test), 'entries.'

# pick a length of the test set
start = 0
end = len(test)-1

# prep for multithreading
tested = [0, 0, 0, 0]
hop = (end-start)/4
threads = []

# get predictions
for t in range(4):#len(output)) : 
	# if you are usuing Gaussian Mixture Model, use getGMMPredictions instead
	# same arguments is fine
	if (t == 3) :
		args = (t, clf, start+t*hop, end+1, feature_list)
	else :
		args = (t, clf, start+t*hop, start+(t+1)*hop, feature_list)

	threads.append(th.Thread(target=getPredictions, args=args))
	threads[-1].start()

# wait for everyone to finish
for t in threads :
	t.join()

# aggregate results
true_pos = np.sum(true_pos, axis=1)
false_pos = np.sum(false_pos, axis=1)
false_neg = np.sum(false_neg, axis=1)

# print table
print '%21s' % "Genre", '%5s' % "t +", '%5s' % "f +", '%5s' % "f -", '%6s' % "prec", '%6s' % "rcl"        
for i in range(len(genres)) :
	den = true_pos[i] + false_pos[i]
	if (den == 0) :
		prec = "n/a"
	else :
		prec = str(1.0 * true_pos[i] / den)[:5]

	den = true_pos[i] + false_neg[i]
	if (den == 0) :
		rcl = "n/a"
	else :
		rcl = str(1.0 * true_pos[i] / den)[:5]

	print '%21s' % genres[i], '%5i' % true_pos[i], '%5i' % false_pos[i], '%5i' % false_neg[i], '%6s' % prec, '%6s' % rcl
