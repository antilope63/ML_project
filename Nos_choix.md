# Nos Choix

Nous avons tester 5 model et leur parametre :

LogisticRegression

SVC

KNeighborsClassifier

RandomForestClassifier

DecisionTreeClassifier

Nous avons finalement choisi LogisticRegression,  
car c’est celui qui obtenait les meilleurs résultats.

Pour mesurer les résultats, nous avons choisi la métrique **f1_macro**.  
C’est la plus pertinente, car elle combine les résultats des vrais positifs, des faux négatifs et des faux positifs.  
Cela permet d’avoir une vision plus générale et de ne pas regarder seulement les bons ou mauvais résultats du modèle.

Si vous voulez plus de détails, pour chaque espèce, on calcule le F1.

$$

F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}
$$

puis on calcul ensuite F1_macro :

$$

F1\_{macro} = \frac{F1\_{classe1} + F1\_{classe2} + F1\_{classe3}}{3}
$$

pour mieux comprendre voici un exemple :

### Espece 1

- TP=40, FP=10, FN=5
  Precision = 40/(40+10)=40/50=0.8
  Recall = 40/(40+5)=40/45≈0.8889
  F1 = 2×0.8×0.8889/(0.8+0.8889) ≈ **0.8421**

### Espece 2

- TP=30, FP=5, FN=15
  Precision = 30/35≈0.8571
  Recall = 30/45≈0.6667
  F1 ≈ **0.75**

### Espece 3

- TP=20, FP=10, FN=10
  Precision = 20/30≈0.6667
  Recall = 20/30≈0.6667
  F1 ≈ **0.6667**

### **F1 macro**

$F1\_{macro} = (0.8421 + 0.75 + 0.6667)/3 \approx 0.7529$
