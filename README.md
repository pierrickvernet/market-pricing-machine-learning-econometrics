# Modélisation et Prédiction des Prix de Marché : Approches Économétriques et Apprentissage Statistique

Ce projet a été réalisé dans le cadre de mon Master 1 IREF (Ingénierie Financière, Recherche et Économétrie) pour le module d'Économétrie des big data. L'objectif principal est d'analyser un jeu de données financières de 25 000 observations et d'évaluer de manière comparative la pertinence de modèles économétriques linéaires classiques et d'algorithmes de Machine Learning pour prédire la variable cible `market_price`.

## 📌 Structure de la Démarche

Le projet est structuré au sein d'un unique notebook Jupyter `pricing.ipynb` découpé en plusieurs phases méthodologiques :

1. **Analyse Descriptive :** Exploration de la structure des données, visualisation de la distribution de la variable dépendante et analyse quantitative des données manquantes.
2. **Préparation des Données :** Évaluation des stratégies de traitement des valeurs manquantes (exclusion par *Listwise Deletion* rejetée car éliminant 92 % des lignes, choix final d'une imputation par la médiane pour les variables numériques et le mode pour les qualitatives) et encodage des variables catégorielles via la création de variables muettes (*Dummification*).
3. **Modélisation Économétrique :** Estimation d'une régression linéaire par les Moindres Carrés Ordinaires (OLS) et interprétation économique des coefficients (impact majeur du risque, de la rentabilité opérationnelle, et primes sectorielles/notations).
4. **Sélection de Variables :** Implémentation et comparaison d'algorithmes de réduction de dimensionnalité pour limiter la multicolinéarité et améliorer la parcimonie :
   * Forward Selection
   * Backward Selection
   * Lasso 
5. **Apprentissage Statistique (Machine Learning) :** Entraînement et optimisation d'hyperparamètres par validation croisée à 10 blocs (*10-fold Cross-Validation*) :
   * Algorithme des K plus proches voisins (*KNN Regressor*)
   * Arbre de décision de régression (*Decision Tree*)
6. **Discussion Finale :** Évaluation comparative des performances sur un échantillon de test indépendant (80% Train / 20% Test) et choix du modèle optimal.

## 📊 Synthèse des Résultats et Performances

Chaque modèle a été évalué sur l'échantillon de test à l'aide de trois métriques clés : le coefficient de détermination ($R^2$), la racine de l'erreur quadratique moyenne (RMSE) et l'erreur absolue moyenne (MAE).

* **Régression Linéaire (OLS post-sélection pas à pas) :** Obtenant un $R^2$ d'environ **0,71**, le modèle linéaire est extrêmement robuste. La procédure de sélection a permis de réduire l'espace à 16 variables clés tout en éliminant la multicolinéarité (chute drastique du *Condition Number*).
* **Arbre de Décision (Profondeur optimale = 5) :** Propose des performances très proches du modèle linéaire ($R^2 \approx 0,71$), prouvant la stabilité de la structure des données.
* **K plus proches voisins (KNN avec K = 25) :** Présente des performances nettement inférieures ($R^2 \approx 0,36$), montrant la relative inefficacité de l'approche par distance sur ce type d'espace géométrique dummifié.

Le modèle de **régression linéaire avec sélection Backward** est retenu comme optimal car il allie un pouvoir prédictif maximal, une grande parcimonie et surtout une parfaite interprétabilité économique des coefficients.

## 🛠️ Technologies et Librairies Utilisées

Le projet est codé en **Python 3** et s'appuie sur l'écosystème scientifique standard :
* **Manipulation de données :** `pandas`, `numpy`
* **Visualisation graphique :** `matplotlib`, `seaborn`
* **Modélisation économétrique :** `statsmodels`
* **Machine Learning & Préprocessing :** `scikit-learn` (`train_test_split`, `StandardScaler`, `SimpleImputer`, `GridSearchCV`, `LassoCV`, `KNeighborsRegressor`, `DecisionTreeRegressor`)
