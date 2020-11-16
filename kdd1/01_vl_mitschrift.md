# 01 Vorlesung
## Organisatorisches

Übungen nur für Fragen
Musterlösungen werden veröffentlicht

## What is Data Mining
Informationen oder Muster aus großen Datenbanken extrahieren.

Descriptive Learning:
Daten besser verstehen
- Pattern Recognition, clustering, outliner detection

Predictive Learning:
Bessere Vorhersagen
- traffic prediction, labeling

Prescriptive Learning:
Bessere Aktionen finden
- predictive maintenance, autonomous driving

Zu viele Daten, haben haber zu wenig Wissen darüber

## Data Mining - Potential Applications
- Market analysis and management
- Risk analysis
- Fraud detection
- Text Mining
- Intelligent query answering

## Knowledge Discovery Process (KDP)
1. Informationen aus unterschiedlichen Databases
  - werden gecleant
2. Data Warehose
  - die Daten, die uns interessieren verändert/transformiert/selektiert
3. Task-relevant data
  - datamining
4. Patterns
  - Visualization/ Evaluation
5. Knowledge

--> Von Datenbanken zum Wissen

## KDP: Data cleaning & Integration
- Größter Aufwand
- Datenbanken werden zusammengefasst
  - Garbage in, Garbadge out? oder Quality-data?

Mögliches Problem:
- Mapping of attribute names:  
  - Kundennummer (C_Nr) könnte anders heißen (obj_id) in anderer Datenbank.
- Inkonsistente Daten, Noise
- Fehlende Werte müssen nachgetragen/berechnet werden

## KDP: Focusing on Task-relevant data
- Task: Nützliche feature finden, Dimensionen/Variablen reduzieren
- Selections: Zeilen/Beispiele auswählen
- Projection: Spalten/attribute Auswählen
- Transformations:
  - Vergröberung/Discretization of numerical attributes

-> Create target data set

## KDP: Basic Data Mining Tasks
-> Find patterns of interest
- Tasks: wie finden wir unser Wissen / haben wir label
  - Supervised, Semi-Supervised, unsupervised Learning

-> Algorithmus auswählen

## Basic Data Mining Tasks:
### Frequent Itemset Mining
Häufig gleichzeitig/coocurring items weisen auf eine Correlation hin.
- "Was wird häufig zusammengekauft?"
- Anwendungsbereiche:
  - Market-basket analysis
  - Cross-marketing, Catalogue design
  - basis for clustering, classification
  - Mögliche Regeln festlegen: Wenn x gekauft wird, dann auch y (Association rule mining)

### Clustering
- Mehrere Objekte mit unbekanntem Label
- Task: wollen ähnliche Ojekte zusammen gruppieren
- Ähnlichkeit wird mit einer "similarity function" beschrieben. Ähnlichkeiten zwischen Clustern sollen möglichst unterschiedlich sein

- Anwendungsbereiche:
  - Customer profiling
  - Document/image collections
  - web access patterns

### Classification
Haben Label aus Training data.
- Task: Modell lernen, um spätere Objekte labeln zu können.
- Label sollen trennbar sein. Nicht immer möglich

- Applications:
  - Krankheitentyp
  - Unbekannte Werte berechnen

### Regression
Zahlen vorhersagen mit Training data

### Generalization Levels
Haben Daten die sehr fein granuliert sind. Wollen es aber garnicht so genau wissen.
-> Daten ersetzen

### Other Methods
- Outliner Detection
- Trends and Evolution analysis

## KDP: Evaluation and Visualization
- Informationsvisualisierung:
  - Daten grafisch darstellen
  - Erkannte Muster darstellen
  - Entscheidungsprozess darstellen
  - interaktive Darstellung (mehr, weniger Cluster)
--> Soll für Verständnis sorgen

===================================================
Data Mining ist der Prozess Information aus Daten zu gewinnen.
- KDP Prozess: data cleaning, integration, selection/transformation, mining, Pattern evaluation, Knowledge presentation/Visualization
- Data mining functionalities: characterization, discrimination, association, classification, clustering, outlier and trend analysis

===================================================

# Vorlesung 2
## Datatypes
TODO: differnet distance functions
  - p-norm
  - euclidean distance
  - manhatten distance
  - maximum distance
  - weighted p-distances

## Generalization
### Metric Space fulfills:
- Symmetry d(q,p) = d(p,q)
- d(p,q) = 0 <=> p=q
All relevant examples are regarded
- triangel inequality d(p,q) < d(p,o) + d(o,q)

### Similarity Queries:
- epsilion range query:
All objects that are within the radius epsilion. So where the distance between o and q is smallerequal to epsilion
   - d(o,q) < epsilion

- Nearest neigbor query:
find the nearest object to q

- k-Nearest neighbor:
find k nearest objects to q

## Categorical Data
no numbers, only symbols
{mathe, sport, deutsch,...}

--> how to compare values?
example : if q == p then 0, else 1

we can also enumerate or build a generalization hierarchy (a decision tree)  ==> get a way to define distances. Are the two objects in the same domain? Path length

## Ordinal Data
there is a order that fulfills:
- transitivity
- antisymmetry
- totality (p ≤ q) or (q ≤ p)
### Examples
- words & lexicographic ordering
- vague sizes
- frequencies

## sequences, vectors, sets
[obvious]

## Complex data types
graphs, shapes, image, audio
==> we need similarity Modells
- feature engineering: handcrafted features
- feature learning: learned by machine learning NNs
- Kernel trick

## Objects and Attributes
- Data tables (Relational Model) == basic Database

## Feature Extraction
"how many pixels are blue?"
- make histogram of color diversity into Vector Space

## Similarity Queries


# Similarity Search
- sequential scan: look at each item and check distances individually O(n)
- Filter refine architecture:
  - get small subset from database
  - calc distance for subset
  - "dimensionality reduction": 3D->2D, 1000D->5D
  --> Project items into 2D space
- Indexing structures:
  - hierarchical indexing techniques

## Indexing
- Organize data so that its easy/fast to get to each object. Can prune subtrees that are not relevant to query
- R-tree cant always guarantee log-behaviour

## Data Reduction
- Daten zusammenfassen/generalisieren
### Strategies
1. Numerosity reduction: Delete objects/examples/rows
2. dimensionality reduction: reduce number of attributes
3. Quantifization, Discretization: reduce number of different values in column

## Data aggregation
Generalization : Age 1-99 -> baby, kid, teen, tween, adult...
Aggregation : Introduce additional attribute. For example counter => weniger beispiele mit gleicher aussagekraft

## Distributive aggregate Measures
## Algebraic aggregate Measures
--> pretty straight forward. sum(D1) + sum(D2)
## Holistic aggregate Measures
--> not as simple: median(D1) and median(D2) is difficult do obtain median(D1 and D2)  
--> efficiency problems

## Measuring Central Tendency
- mean, weights can be added
- mid-range: maximum value + minimum value divided by 2. Builds the average
- median: take middle vlaue, or average of the two middle values. only ordinal data. We need an order (holistic measure)

## Measuring the Dispersion of data
- Variance: measure the spread around the mean. is 0 when all values are equal
(Algebraic measure)


























asdf
