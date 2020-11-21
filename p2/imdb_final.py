import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from time import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# path_train_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/pos'
# path_train_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/train/neg'
# path_test_pos = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/pos'
# path_test_neg = 'C:/Users/linsh/PycharmProjects/Comp551p2/aclImdb/test/neg'

path_train_pos = '/Users/tianchima/Desktop/aclImdb/train/pos' 
path_train_neg = '/Users/tianchima/Desktop/aclImdb/train/neg' 
path_test_pos = '/Users/tianchima/Desktop/aclImdb/test/pos' 
path_test_neg = '/Users/tianchima/Desktop/aclImdb/test/neg' 


def read_data(path):
    files = os.listdir(path)
    data = []
    files.sort()
    i = 0
    for file in files:
        #if i > 100:
         #   break
        i = i+1
        if not os.path.isdir(file):
            with open(path + '/' + file, 'r', encoding='UTF-8') as f:
                data.append(f.read())
    return data


def get_dataset():
    test_pos_data = read_data(path_test_pos)
    test_neg_data = read_data(path_test_neg)
    train_data = []
    train_label = []
    test_label = []
    train_pos_data = read_data(path_train_pos)
    train_neg_data = read_data(path_train_neg)
    for i in range(len(train_pos_data)):
        train_data.append(train_pos_data[i])
        train_label.append(1)
        train_data.append(train_neg_data[i])
        train_label.append(-1)
    test_data = test_pos_data + test_neg_data
    for i in range(len(test_pos_data)):
        test_label.append(1)
    for i in range(len(test_neg_data)):
        test_label.append(-1)

    stemmer = PorterStemmer()
    stop_words = list(stopwords.words('english'))
    for i in range(len(train_data)):
        #train_data[i] = re.sub(r'[^A-z ]', '', train_data[i])
        word_list = word_tokenize(train_data[i])
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        train_data[i] = ''
        for j in range(len(word_list)):
            #if word_list[j] not in stop_words:
                #if len(word_list[j])<20:
            train_data[i] = train_data[i] + word_list[j] + ' '

    for i in range(len(test_data)):
        #test_data[i] = re.sub(r'[^A-z ]', '', test_data[i])
        word_list = word_tokenize(test_data[i])
        for j in range(len(word_list)):
            word_list[j] = (stemmer.stem(word_list[j]))
        test_data[i] = ''
        for j in range(len(word_list)):
            #if word_list[j] not in stop_words:
                #if len(word_list[j])<20:
            test_data[i] = test_data[i] + word_list[j] + ' '

    return train_data, test_data, train_label, test_label





def svm(dataset):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LinearSVC(random_state=0))])
    param_dist = {'clf__dual': [True, False],
              'clf__loss': ['hinge', 'squared_hinge'],
              'clf__C': loguniform(1e-3, 1e3),
              'clf__tol': loguniform(1e-11, 1e-4),
              'clf__fit_intercept': [True, False],
              'vect__min_df': loguniform(1e-4, 1e-2), 
              'vect__max_df': np.linspace(0.5,0.9,num=10, dtype = float),
              'vect__stop_words':[None, 'english'],
              'vect__token_pattern': ['\w{2,}','\w{1,}'],
              'vect__ngram_range' : [(1,2),(1,1)]}
    n_iter_search = 200
    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, n_jobs=-1) 
    

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    with open('/Users/tianchima/Desktop/Trial1/svm.txt', 'w') as f:
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print ("Parameters: {0}".format(results['params'][candidate]))
            f.write('Mean validation score: '+str(results['mean_test_score'][candidate])+ '\n')
            f.write('std: '+str(results['std_test_score'][candidate])+ '\n')
            f.write('Parameters: '+str(results['params'][candidate])+'\n')
        test_label_predict = random_search.best_estimator_.predict(test_data)
        accruracy = accuracy_score(test_label, test_label_predict)
        print(results)
        print('accruracy = ',accruracy)

        f.write('accruracy = ' + str(accruracy)+'\n'+'\n')
        f.write(str(random_search.best_params_)+'\n'+'\n')
        f.write(str(results))


    return random_search.best_estimator_


def LR(dataset):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression(random_state=0))])
    param_dist = {'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'clf__dual': [True, False],
              'clf__C': loguniform(1e-3, 1e3),
              'clf__tol': loguniform(1e-11, 1e-4),
              'clf__fit_intercept': [True, False],
              'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'clf__max_iter': stats.randint(50,200),
              'clf__warm_start': [True, False],
              'clf__multi_class': ['auto', 'ovr', 'multinomial'],
              'clf__l1_ratio': stats.uniform(0,1),
              'vect__min_df': loguniform(1e-4, 1e-2),
              'vect__max_df': np.linspace(0.5,0.9,num=10, dtype = float),
              'vect__stop_words':[None, 'english'],
              'vect__token_pattern': ['\w{2,}','\w{1,}'],
              'vect__ngram_range' : [(1,2),(1,1)]}

    n_iter_search = 200
    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, n_jobs=-1) 
    

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    with open('/Users/tianchima/Desktop/Trial1/LR.txt', 'w') as f:
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print ("Parameters: {0}".format(results['params'][candidate]))
            f.write('Mean validation score: '+str(results['mean_test_score'][candidate])+ '\n')
            f.write('std: '+str(results['std_test_score'][candidate])+ '\n')
            f.write('Parameters: '+str(results['params'][candidate])+'\n')
        test_label_predict = random_search.best_estimator_.predict(test_data)
        accruracy = accuracy_score(test_label, test_label_predict)
        print(results)
        print('accruracy = ',accruracy)

        f.write('accruracy = ' + str(accruracy)+'\n'+'\n')
        f.write(str(random_search.best_params_)+'\n'+'\n')
        f.write(str(results))

    return random_search.best_estimator_

def AdaB(dataset):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', AdaBoostClassifier(random_state=0))])
    param_dist = {'clf__learning_rate':  stats.uniform(0, 1),
              'clf__n_estimators': stats.randint(10,400),
              'clf__algorithm': ['SAMME', 'SAMME.R'],
              'clf__base_estimator': [None, LinearSVC(), LogisticRegression(), MultinomialNB()],
              'vect__min_df': loguniform(1e-4, 1e-2),
              'vect__max_df': np.linspace(0.5,0.9,num=10, dtype = float),
              'vect__stop_words':[None, 'english'],
              'vect__token_pattern': ['\w{2,}','\w{1,}'],
              'vect__ngram_range' : [(1,2),(1,1)]}
              
    n_iter_search = 200
    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, n_jobs=-1) 
    

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    with open('/Users/tianchima/Desktop/Trial1/AdaB.txt', 'w') as f:
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print ("Parameters: {0}".format(results['params'][candidate]))
            f.write('Mean validation score: '+str(results['mean_test_score'][candidate])+ '\n')
            f.write('std: '+str(results['std_test_score'][candidate])+ '\n')
            f.write('Parameters: '+str(results['params'][candidate])+'\n')
        test_label_predict = random_search.best_estimator_.predict(test_data)
        accruracy = accuracy_score(test_label, test_label_predict)
        print(results)
        print('accruracy = ',accruracy)

        f.write('accruracy = ' + str(accruracy)+'\n'+'\n')
        f.write(str(random_search.best_params_)+'\n'+'\n')
        f.write(str(results))
 

    return random_search.best_estimator_

def DecisionT(dataset):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', DecisionTreeClassifier(random_state=0))])
    param_dist = {'clf__criterion': ['gini', 'entropy'],
              'clf__splitter': ['best', 'random'],
              'clf__max_depth': np.arange(2,50, dtype = int).tolist()+[None],
              'clf__min_samples_split': loguniform(1e-4, 1e-1),
              'clf__min_samples_leaf': stats.randint(1,200),
              'clf__max_features': np.linspace(0.01,1,num = 10, dtype=float).tolist()+['auto', 'sqrt', 'log2', None],
              'clf__max_leaf_nodes': [None]+ (np.power(10, np.arange(1,4,step =0.5)).astype(np.int)).tolist(),
              'clf__min_impurity_decrease': [0.0]+np.array(np.power(10, np.arange(-10,-4,step =0.5)),dtype =float).tolist(),
              'vect__min_df': loguniform(1e-4, 1e-2),
              'vect__max_df': np.linspace(0.5,0.9,num=10, dtype = float),
              'vect__stop_words':[None, 'english'],
              'vect__token_pattern': ['\w{2,}','\w{1,}'],
              'vect__ngram_range' : [(1,2),(1,1)]}
              
    n_iter_search = 200
    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, n_jobs=-1) 
    

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    with open('/Users/tianchima/Desktop/Trial1/DecisionT.txt', 'w') as f:
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print ("Parameters: {0}".format(results['params'][candidate]))
            f.write('Mean validation score: '+str(results['mean_test_score'][candidate])+ '\n')
            f.write('std: '+str(results['std_test_score'][candidate])+ '\n')
            f.write('Parameters: '+str(results['params'][candidate])+'\n')
        test_label_predict = random_search.best_estimator_.predict(test_data)
        accruracy = accuracy_score(test_label, test_label_predict)
        print(results)
        print('accruracy = ',accruracy)

        f.write('accruracy = ' + str(accruracy)+'\n'+'\n')
        f.write(str(random_search.best_params_)+'\n'+'\n')
        f.write(str(results))


    return random_search.best_estimator_

def RandomF(dataset):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', RandomForestClassifier(random_state=0))])
    param_dist = {'clf__n_estimators': np.array(np.power(10, np.arange(1,3,step =0.1)),dtype = int),
              'clf__criterion': ['gini', 'entropy'],
              'clf__max_depth': np.arange(2,50, dtype = int).tolist()+[None],
              'clf__min_samples_split': loguniform(1e-4, 1e-2),
              'clf__min_samples_leaf': stats.randint(1,200),
              'clf__max_features': np.linspace(0.01,1,num = 10, dtype=float).tolist()+['auto', 'sqrt', 'log2', None],
              'clf__max_leaf_nodes': [None]+ (np.power(10, np.arange(1,4,step =0.5)).astype(np.int)).tolist(),
              'clf__min_impurity_decrease': [0.0]+np.array(np.power(10, np.arange(-10,-4,step =0.5)),dtype = float).tolist(),
              'clf__bootstrap': [True, False],
              'clf__oob_score': [True, False],
              'clf__warm_start':[True, False],
              'clf__max_samples': stats.uniform(0,1),
              'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
              'clf__verbose': [True, False],
              'clf__min_weight_fraction_leaf': loguniform(1e-4, 1e-1),
              'vect__min_df': loguniform(1e-4, 1e-2),
              'vect__max_df': np.linspace(0.5,0.9,num=10, dtype = float),
              'vect__stop_words':[None, 'english'],
              'vect__token_pattern': ['\w{2,}','\w{1,}'],
              'vect__ngram_range' : [(1,2),(1,1)]}
              
    n_iter_search = 200
    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, n_jobs=-1) 
    

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    with open('/Users/tianchima/Desktop/Trial1/RandomF.txt', 'w') as f:
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print ("Parameters: {0}".format(results['params'][candidate]))
            f.write('Mean validation score: '+str(results['mean_test_score'][candidate])+ '\n')
            f.write('std: '+str(results['std_test_score'][candidate])+ '\n')
            f.write('Parameters: '+str(results['params'][candidate])+'\n')
        test_label_predict = random_search.best_estimator_.predict(test_data)
        accruracy = accuracy_score(test_label, test_label_predict)
        print(results)
        print('accruracy = ',accruracy)

        f.write('accruracy = ' + str(accruracy)+'\n'+'\n')
        f.write(str(random_search.best_params_)+'\n'+'\n')
        f.write(str(results))


    return random_search.best_estimator_

def Bag(dataset, estimator1, estimator2, estimator3,estimator4,estimator5):

    vect_and_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', BaggingClassifier(random_state=0))])
    param_dist = {'clf__base_estimator': [None, estimator1, estimator2, estimator3, estimator4, estimator5,MultinomialNB()],
              'clf__n_estimators': stats.randint(10,400),
              'clf__max_features': stats.uniform(0, 1),
              'clf__max_samples': stats.uniform(0, 1),
              'clf__bootstrap': [True, False],
              'clf__bootstrap_features': [True, False],
              'clf__oob_score': [True, False],
              'clf__warm_start': [True, False],
              'vect__min_df': loguniform(1e-4, 1e-2),
              #(np.power(10, np.arange(0,3,step =0.5)).astype(np.int)),
              'vect__max_df': np.linspace(0.5,0.9,num=10, dtype = float),
              'vect__stop_words':[None, 'english'],
              'vect__token_pattern': ['\w{2,}','\w{1,}'],
              'vect__ngram_range' : [(1,2),(1,1)]}
              
    n_iter_search = 200
    random_search = RandomizedSearchCV(vect_and_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, n_jobs=-1) 
    

    train_data, test_data, train_label, test_label = get_dataset()
    start = time()
    random_search.fit(train_data, train_label)
    print("RandomizedSearchCV took %.2f seconds" % ((time() - start)))
    results = random_search.cv_results_
    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    with open('/Users/tianchima/Desktop/Trial1/Bag.txt', 'w') as f:
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print ("Parameters: {0}".format(results['params'][candidate]))
            f.write('Mean validation score: '+str(results['mean_test_score'][candidate])+ '\n')
            f.write('std: '+str(results['std_test_score'][candidate])+ '\n')
            f.write('Parameters: '+str(results['params'][candidate])+'\n')
        test_label_predict = random_search.best_estimator_.predict(test_data)
        accruracy = accuracy_score(test_label, test_label_predict)
        print(results)
        print('accruracy = ',accruracy)

        f.write('accruracy = ' + str(accruracy)+'\n'+'\n')
        f.write(str(random_search.best_params_)+'\n'+'\n')
        f.write(str(results))


    return


if __name__ == '__main__':

    dataset = get_dataset()

    estimator1 = svm(dataset)
    estimator2 = LR(dataset)
    estimator4 = DecisionT(dataset)
    estimator5 = RandomF(dataset)
    estimator3 = AdaB(dataset)
    Bag(dataset, estimator1, estimator2, estimator3, estimator4, estimator5)