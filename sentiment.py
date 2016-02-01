import sys
import collections
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import nltk
import random
random.seed(0)
import itertools
from collections import defaultdict
import operator
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    train_pos_words = []
    train_neg_words = []
    test_pos_words = []
    test_neg_words = []
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [],[],[],[]
    
    train_pos_dict = {}
    train_neg_dict = {}
    
    #remove duplicates within each review/tweet
    i=0
    while i<len(train_pos):
        train_pos[i] = list(set(train_pos[i])-stopwords)
        i+=1
    
    i=0
    while i<len(train_neg):
        train_neg[i] = list(set(train_neg[i])-stopwords)
        i+=1
    
    i=0
    while i<len(test_pos):
        test_pos[i] = list(set(test_pos[i])-stopwords)
        i+=1
    
    i=0
    while i<len(test_neg):
        test_neg[i] = list(set(test_neg[i])-stopwords)
        i+=1
    
    
    #flatmap all the data in one file        
    train_pos_list = list(itertools.chain(*train_pos))
    train_neg_list = list(itertools.chain(*train_neg))
    test_pos_list = list(itertools.chain(*test_pos))
    test_neg_list = list(itertools.chain(*test_neg))
    
    total_train_list = list(set(train_pos_list + train_neg_list))
    
    #create dictionary for frequency of words
    train_pos_dict = defaultdict( int )
    for w in train_pos_list:
        train_pos_dict[w] += 1
    
    #create dictionary for frequency of words
    train_neg_dict = defaultdict( int )
    for w in train_neg_list:
        train_neg_dict[w] += 1
    
    
    train_features = filter(lambda x: count_check(x,train_pos_dict,train_neg_dict,len(train_pos),len(train_neg)),total_train_list)
    
    
    
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    for item in train_pos:
        train_pos_vec.append(map(lambda x: 1 if x in item else 0,train_features))
    
    for item in train_neg:
        train_neg_vec.append(map(lambda x: 1 if x in item else 0,train_features))
    
    for item in test_pos:
        test_pos_vec.append(map(lambda x: 1 if x in item else 0,train_features))
    
    for item in test_neg:
        test_neg_vec.append(map(lambda x: 1 if x in item else 0,train_features))
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def count_check(word, pos, neg, len_pos, len_neg):
    count_pos = pos[word]
    count_neg = neg[word]
    if (float(count_pos)/len_pos>=0.01 or float(count_neg)/len_neg>=0.01) and (count_pos>=2*count_neg or count_neg>=2*count_pos):
        return True
    else:
        return False

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = labelizeReviews(train_pos, 'TR_P')
    labeled_train_neg = labelizeReviews(train_neg, 'TR_N')
    labeled_test_pos = labelizeReviews(test_pos, 'T_P')
    labeled_test_neg = labelizeReviews(test_neg, 'T_N')
    
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    def getDocvecs(model, reviews,labelPrefix):
        vectors = []
        for i,v in enumerate(reviews):
            vectors.append(model.docvecs[labelPrefix+str(i)])
        return vectors    
    
    train_pos_vec = getDocvecs(model,train_pos,'TR_P')
    train_neg_vec = getDocvecs(model,train_neg,'TR_N')
    test_pos_vec = getDocvecs(model,test_pos,'T_P')
    test_neg_vec = getDocvecs(model,test_neg,'T_N')
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def labelizeReviews(reviews, label_type):
    labelized = []
    i=0
    for v in reviews:
        labelized.append(LabeledSentence(v, [label_type+str(i)]))
        i+=1
    return labelized

def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    clf = BernoulliNB(alpha=0.1,binarize=None)
    nb_model = clf.fit(train_pos_vec+train_neg_vec, Y)
    
    cll = LogisticRegression()
    lr_model = cll.fit(train_pos_vec+train_neg_vec, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    cll = LogisticRegression()
    lr_model = cll.fit(train_pos_vec+train_neg_vec, Y)
    
    clf = GaussianNB()
    nb_model = clf.fit(train_pos_vec+train_neg_vec, Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    y_pred = model.predict(test_pos_vec + test_neg_vec)
    y_true = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)
    conf_mat =  confusion_matrix(y_true,y_pred)
    count=0
    i=0

    tn = conf_mat[0][0]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    tp = conf_mat[1][1]
    accuracy = float(tp+tn)/(tp+tn+fp+fn)
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
