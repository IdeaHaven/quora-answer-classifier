from sklearn import linear_model, cross_validation, metrics
from scipy import sparse, array
from itertools import combinations

import numpy as np
import pandas as pd
import sklearn

SEED = 25

# === THIS WORKS GREAT === #
def load_data(filename):
    train = pd.read_csv(filename, sep="\s", skiprows=1, names=range(25), nrows=4499)
    test = pd.read_csv(filename, sep="\s", skiprows=4500, names=range(25))
    print ('munging the data')    
    for i in range(2,11):
        train[i] = train[i].map(lambda i: float(str(i)[2:]))
    for j in range(11,25):
        train[j] = train[j].map(lambda i: float(str(i)[3:]))       
    labels = train[1]
    # data should really be [:,[0,2,3....]], but [0] is ugly
    data = train.loc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]    
    return labels, data # labels = y, data = X

# === SUPER BROKEN === #    
def group_data(data, degree=3):
    new_data = []
    m,n = data.shape
    # something about this is significantly messing up dt/dp
    for indices in combinations(range(n), degree):
        #print(indices, "\t", data[:,indices])
        new_data.append([data.loc[:,indices]])
    return array(new_data).T
    
# === THIS WORKS FINE === #    
def save_results(predictions, filename):
    content = ['id,ACTION']
    for i, p in enumerate(predictions):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print ('data saved')


# === Something is causing each iteration to return identical results to the previous, but I suspect cv_loop itself to be functioning fine === #

# Output:
# AUC (fold 1/10): 0.526255
# AUC (fold 2/10): 0.533174
# AUC (fold 3/10): 0.538685
# AUC (fold 4/10): 0.513203
# AUC (fold 5/10): 0.520270
# AUC (fold 6/10): 0.566535
# AUC (fold 7/10): 0.535882
# AUC (fold 8/10): 0.446576
# AUC (fold 9/10): 0.454598
# AUC (fold 10/10): 0.496158
# C: 10.767202 Mean AUC: 0.513133

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit_transform(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

def main(args):
    print ('loading the data')
    y, X = load_data(args['train'])
    
    print ('setting the logistic model')
    model = linear_model.LogisticRegression(C=3)
    
    print ('transforming data')
    num_train = np.shape(X)[0]
    
    # these output a failbus
    dp = group_data(X, degree=2) # correct length of [(23*22)/2], but outputs '<generator object <genexpr> at 0x24E3EF08>' 
    dt = group_data(X, degree=3)
    
    # busted dp/dt results in busted X_2/X_3    
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]

    X_train_all = np.hstack((X, X_2, X_3))
    num_features = X_train_all.shape[1]
    
    print ('performing greedy feature selection')
    score_hist = []
    N = 10 # number of cv_loop iterations
    good_features = set([])
    # Greedy feature selection loop
    while len(score_hist) < 3 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(2,24):#X.columns:
            if f not in good_features:
                feats = list(good_features) + [f]
                # the fundamental difference between this and last night's code is that the data set is NOT sparse, which means model.fit does not work and sparse.hstack/tocsr() is useless. I suspect this to be a major reason for problems
                #Xt = sparse.hstack([X[j] for j in feats]).tocsr()
                #Xt = np.hstack(X[j] for j in feats)
                score = cv_loop(X, y, model, N) # this x is always the same
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
        
    print ('making predictions and saving data to ', args['submit'])
    predictions = model.predict_proba(X)[:,1]
    save_results(predictions, args['submit'])

# This is where everything starts    
if __name__ == '__main__':
    args = { 'train': 'input00.txt',
             'submit': 'output00.txt' }
    main(args)