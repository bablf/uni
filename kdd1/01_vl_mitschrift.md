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
