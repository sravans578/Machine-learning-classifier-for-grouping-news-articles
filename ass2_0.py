#Loadind Data from fetch_20newsgroups :https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
from sklearn.datasets import fetch_20newsgroups

#Loading a subsection of categories
sub_cat = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
data = fetch_20newsgroups(subset='all', categories=sub_cat)

list(data.target_names)
data.filenames.shape
data.target.shape
data.target[:10]
#print (type(data))
#renaming data to corpus
corpus = data
corpus_new  = data
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
stop_words=set(stopwords.words("english"))

#using regex to remove unwanted characters
import re
# nltk.download('punkt')
print(corpus.data[0])

stem = PorterStemmer()
lem = WordNetLemmatizer()


list_of_articles = []
list_of_articles_NN = []

#Cleaning data (Without POS Tagging)
for article in corpus.data:
    article = re.sub('[^a-zA-Z]',' ',article)
    article = article.lower()
    words = word_tokenize(article)
    #words = [lem.lemmatize(word,"v")for word in words if not word in stop_words]
    words = [stem.stem(word)for word in words if not word in stop_words]
    clean_sentence =' '.join(words)
    list_of_articles.append(clean_sentence)
    
    
#Cleaning data (With POS Tagging)
for article in corpus_new.data:
    words_1 = word_tokenize(article)
    tags = nltk.pos_tag(words_1)
    new_list = []
    for obj in tags:       
        if(obj[1]in ["NN","NNS","NNP","NNPS"] ):
            new_list.append(obj[0])
    article_with_NN = ' '.join(new_list)
    new_list.clear()
    article_with_NN = re.sub('[^a-zA-Z]',' ',article_with_NN)
    article_with_NN = article_with_NN.lower()
    words = word_tokenize(article_with_NN)
    #words = [lem.lemmatize(word,"v")for word in words if not word in stop_words]
    words = [stem.stem(word)for word in words if not word in stop_words]
    clean_sentence =' '.join(words)
    list_of_articles_NN.append(clean_sentence)


  
    
#Count vector (with POS)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_new = vectorizer.fit_transform(list_of_articles_NN).toarray()

#Count vector (without POS)

X = vectorizer.fit_transform(list_of_articles).toarray()

#TFIDF calculation (Without POS)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
Y = corpus.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, Y, test_size=0.3, random_state=123)

from sklearn import svm
from sklearn.metrics import  precision_score, recall_score,f1_score,accuracy_score
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred =clf.predict(X_test)
print("Precision: %1.3f" % precision_score(y_test, y_pred,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred) )

#Confusion matrix for SVM
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test, y_pred)

#Confusion matrix for Naivebayes
from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
confusion_matrix(y_test, y_pred2)
print("Precision: %1.3f" % precision_score(y_test, y_pred2,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred2,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred2,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred2) )

#SVM with linear kernel
svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred3 =svclassifier.predict(X_test)
print("Precision: %1.3f" % precision_score(y_test, y_pred3,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred3,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred3,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred3) )
classification_report(y_test,y_pred3)

#SVC with Gaussian kernel
svc_G = svm.SVC(kernel='rbf') 
svc_G.fit(X_train, y_train)
y_pred4 =svc_G.predict(X_test)
print("Precision: %1.3f" % precision_score(y_test, y_pred4,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred4,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred4,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred4) )

##########################################################################
#TFIDF calculation (With POS)

transformer = TfidfTransformer(smooth_idf=False)
tfidf_new = transformer.fit_transform(X_new)

X_train, X_test, y_train, y_test = train_test_split(tfidf_new, Y, test_size=0.3, random_state=123)

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred =clf.predict(X_test)
print("Precision: %1.3f" % precision_score(y_test, y_pred,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred) )

#Confusion matrix for SVM
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test, y_pred)

#Confusion matrix for Naivebayes
from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
confusion_matrix(y_test, y_pred2)
print("Precision: %1.3f" % precision_score(y_test, y_pred2,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred2,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred2,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred2) )

#SVM with linear kernel
svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred3 =svclassifier.predict(X_test)
print("Precision: %1.3f" % precision_score(y_test, y_pred3,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred3,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred3,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred3) )
classification_report(y_test,y_pred3)

#SVC with Gaussian kernel
svc_G = svm.SVC(kernel='rbf') 
svc_G.fit(X_train, y_train)
y_pred4 =svc_G.predict(X_test)
print("Precision: %1.3f" % precision_score(y_test, y_pred4,average='weighted'))
print("Recall: %1.3f" % recall_score(y_test, y_pred4,average='weighted'))
print("F1: %1.3f\n" % f1_score(y_test, y_pred4,average='weighted'))
print( "accuracy_score:%1.3f\n"%accuracy_score(y_test, y_pred4) )
