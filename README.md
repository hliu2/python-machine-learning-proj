# python machine learning project 
## Introduction 
* Try to predict whether a bank client will subscribe a bank term deposit. 
* Conducted Random Forest for feature selection and data preprocessing 
* Applied several Machine Learning algorithms - Decision Tree, SVM, KNN, Logistic Regression, LDA, and QDA - to predict whether clients will subscribe the bank term deposit
* Obtained an optimal model with a remarkable 72.8% testing accuracy  

## Hightlights 
* Data is unbalanced, 5% of all people will buy this product. If random guess, accuracy should around 5%, this model increase the accuracy to 72.8%
* Defined a f2 score to tune parameters. This criterion doesn't care about hom many True Negative(accurately discriminate thoese people who will not buy this product), but try to maximum the number of True Positive(accurately discriminate thoese people who will buy this product) and minimize False Negative (wrongly predicte people who actually will buy this product as not) and False Positive (wrong predict people who actually will not buy this product as will buy). But among these two types of errors, this criterion penalize more on False Negative than False Positve.    
