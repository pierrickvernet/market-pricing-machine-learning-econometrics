# Modélisation et Prédiction des Prix de Marché : Approches Économétriques et Apprentissage Statistique

## 1. INTRODUCTION
- Cadre de réalisation : Projet réalisé dans le cadre de mon Master 1 IREF pour le module d'Économétrie des Big Data.
- Présentation du sujet : L'objectif de cette étude est de modéliser et de prédire la variable financière `market_price` en évaluant de manière comparative des méthodes économétriques linéaires standards et des algorithmes de Machine Learning non linéaires.
- Cas d'usage : Évaluation d'actifs, tarification financière et aide à la décision quantitativiste à travers la recherche d'un compromis optimal entre puissance prédictive, parcimonie et explicabilité des variables explicatives.

## 2. SOURCES ET DONNÉES
- Périmètre : Dataset financier comprenant 25 000 observations.
- Variable cible : `market_price` (variable continue représentant la valorisation financière ou le prix de marché).
- Variables explicatives : Ensemble de variables financières numériques et catégorielles reflétant le niveau de risque, la rentabilité opérationnelle, le secteur d'activité et la notation financière (credit rating).

## 3. MÉTHODOLOGIE ET DÉTAILS TECHNIQUES
- Traitement des données et Feature Engineering :
  - Analyse quantitative des valeurs manquantes. Le rejet de la méthode d'élimination systématique (*Listwise Deletion*) s'est imposé, celle-ci entraînant la perte de 92 % des observations.
  - Imputation par la médiane pour les variables numériques continuent et par le mode pour les variables catégorielles.
  - Encodage des variables qualitatives par création de variables muettes (*Dummification* / One-Hot Encoding).
  - Normalisation des données via `StandardScaler` pour les algorithmes sensibles aux distances.
- Économétrie et Modélisation :
  - Estimation d'une régression linéaire multiple par les Moindres Carrés Ordinaires (MCO / OLS) à des fins d'interprétation économique des coefficients (mesure des impacts du risque, de la rentabilité et des primes sectorielles).
  - Algorithmes de sélection de variables et réduction de dimension : *Forward Selection*, *Backward Selection* et régularisation L1 (*Lasso CV*) afin d'optimiser la parcimonie et de traiter la multicolinéarité.
- Machine Learning et Optimisation :
  - Séparation du jeu de données en un échantillon d'entraînement (80 %) et un échantillon de test indépendant (20 %).
  - Optimisation des hyperparamètres par recherche sur grille avec validation croisée à 10 blocs (*10-Fold Cross-Validation* via `GridSearchCV`).
  - Algorithmes évalués : Régressur des K plus proches voisins (*KNN Regressor*) et Arbre de décision pour la régression (*Decision Tree Regressor*).
- Stack Technologique :
  - Langage : Python 3
  - Traitement de données : `pandas`, `numpy`
  - Modélisation Économétrique : `statsmodels`
  - Machine Learning & Preprocessing : `scikit-learn` (`train_test_split`, `StandardScaler`, `SimpleImputer`, `GridSearchCV`, `LassoCV`, `KNeighborsRegressor`, `DecisionTreeRegressor`)
  - Visualisation : `matplotlib`, `seaborn`

## 4. CONCLUSION ET RÉSULTATS CLÉS
- Évaluation des performances sur l'échantillon de test (Métriques : $R^2$, RMSE, MAE) :
  - Régression Linéaire (OLS post-sélection Backward) : $R^2 \approx 0,71$. Le modèle conserve 16 variables explicatives clés, élimine la multicolinéarité (baisse marquée du *Condition Number*) et offre une interprétabilité économique intégrale des coefficients.
  - Arbre de Décision (Profondeur optimale = 5) : $R^2 \approx 0,71$. Capture efficacement les structures non linéaires tout en égalant les performances de la régression linéaire.
  - K-Nearest Neighbors (K = 25) : $R^2 \approx 0,36$. Performance nettement inférieure, illustrant la faible efficacité des métriques de distance géométrique dans un espace à forte dimensionnalité comportant de nombreuses variables muettes.
- Modèle Retenu : Le modèle linéaire retenu est la régression OLS associée à une sélection pas à pas *Backward*. Ce choix garantit un pouvoir prédictif maximal, une grande sobriété paramétrique et une explicabilité économique indispensable en ingénierie financière.
- Limites et Perspectives :
  - Tester des architectures d'apprentissage d'ensemble (*Random Forest*, *XGBoost*, *LightGBM*) pour vérifier la présence d'interactions complexes non capturées par l'arbre simple.
  - Approfondir le traitement de la dimensionnalité par des méthodes d'analyse en composantes principales (ACP) ou de régression sur composantes PLS.

## 5. STRUCTURE DU DÉPÔT
```text
.
├── data/
│   └── finance_dataset_25000_with_qualitative.csv   # Jeu de données financier (25 000 observations)
├── notebook/
│   └── pricing.ipynb                                # Notebook Jupyter comprenant l'intégralité du pipeline
└── README.md                                        # Documentation du projet
```

## 6. INSTALLATION ET REPRODUCTION
- Prérequis : Python 3.8 ou supérieur installé sur votre système.
- Étape 1 : Cloner le dépôt GitHub
```bash
git clone [https://github.com/votre-utilisateur/projet-econometrie-pricing.git](https://github.com/votre-utilisateur/projet-econometrie-pricing.git)
cd projet-econometrie-pricing
```
- Étape 2 : Créer et activer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/macOS
# venv\Scripts\activate   # Sur Windows
```
- Étape 3 : Installer les dépendances requises
```bash
pip install -r requirements.txt
```
- Étape 4 : Exécuter l'analyse
Lancer Jupyter Notebook ou Jupyter Lab pour exécuter le traitement complet :
```bash
jupyter lab notebooks/pricing.ipynb
```
