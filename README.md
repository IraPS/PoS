# README

Taggers used: Mystem, TreeTagger, Marmot, Hunpos

Ruscorpora PoS markup was taken as a golden standard

---

Training set: len = 17000 words, good_answers = 0.70 (answers like data : [X,X,X,X] -> label : [X], where X is a PoS tag)

Test set: len = 499 words, good_answers = 0.69 (answers like data : [X,X,X,X] -> label : [X], where X is a PoS tag)

---

--SVC-- (models/mymodel_SVC.pkl)
>accuracy = 0.9879 

--DecisionTreeClassifier-- (models/mymodel_DT.pkl)
>accuracy = 0.9859 

--GaussianNB-- (models/mymodel_NB.pkl)
>accuracy = 0.9418 

--NER-- (models/NER)
>accuracy = 0.9299
