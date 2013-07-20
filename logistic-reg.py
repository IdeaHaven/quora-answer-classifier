import numpy as np
import pandas as pd

from numpy import array
from sklearn import metrics, cross_validation, linear_model
from itertools import combinations
  
def group_data(data, degree=3):
    new_data = []
    m,n = data.shape
    # something about this is significantly messing up dt/dp
    for indices in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indices]])
    return array(new_data).T
       
def save_results(predictions, filename):
    content = ['id,ACTION']
    for i, p in enumerate(predictions):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print ('data saved')

SEED = 25

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

def main():
    print ('loading the data')
    train = pd.read_csv('./train-plucked.csv') # pull version with outliers removed
    test = pd.read_csv('./test.csv')
    y = train.ix[:,24] # last column in array is target data
    X = train.ix[:,1:24] # all cols through 23 are ids or features
    test = test.ix[:,1:24]
    all_data = np.vstack((X, test))
    num_train = np.shape(X)[0] #this is the number of rows in training data
    
    print ('setting the logistic model')
    model = linear_model.LogisticRegression() # no need to use C as data has been scaled
    
    print ('transforming data')
    #groups with additional features
#    dp = group_data(all_data, degree=1)
#    dt = group_data(all_data, degree=1)
    
    #split additional features between train and test sets
    X = all_data[:num_train]
#    X_2 = dp[:num_train]
#    X_3 = dt[:num_train]
    X_test = all_data[num_train:]
#    X_test_2 = dp[num_train:]
#    X_test_3 = dt[num_train:]

    X_test_all = X_test
    X_train_all = X
#    num_features = X_train_all.shape[1]

    print ('performing greedy feature selection')
    score_hist = []
    N = 10 # number of cv_loop iterations
    good_features = set([])
    # Greedy feature selection loop
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(X_train_all.shape[1]):
            if f not in good_features:
#                feats = list(good_features) + [f]              
                score = cv_loop(X_train_all, y, model, N)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features)) 
        
    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    print "Selected features %s" % good_features
    
    print ('performing hyperparameter selection')
    # Hyperparameter selection loop
    score_hist = []
    #Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    Cvals = np.logspace(-4, 4, 15, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(X, y, model, N)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value: %f" % (bestC)    

    print "Training full model..."
    model.fit(X, y)
        
    print ('making predictions and saving data')
    predictions = model.predict_proba(X)[:,1]
    save_results(predictions, "./linregpred.csv")

# This is where everything starts    
if __name__ == '__main__':
    main()