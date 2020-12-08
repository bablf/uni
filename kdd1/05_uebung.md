# Übung 5

## 5-1
d = 1:
Im eindimensionalen Raum (Linie) werden 3 Punkte benötigt, damit man sie nicht
mehr linear separieren kann.

Man nehme an A und C gehören zur Klasse 1 und B zu Klasse -2:
Es kann keine Linie gezeichnet werden, um die Punkte zu separieren

----------------A-------B----------C-------------->

d = 2:
Es werden 4 Punkte benötigt. Siehe bekanntes XOR Problem.



## 5-1
Entropy(T)−k∑i=1 pilog2
minimize:  m∑i=1|Ti|/|T| * entropy(Ti)
Time
1-2: = 0,918
2-7: = 0,918
_>7: = 1
==> 0.94

Gender
m: 0,971
f: 0,918
==> 0,95

Area
urban: = 0
rual = 0,722
==> 0,45

Daher Area als erste aufspaltung
Dann nochmal Time und Gender berechnen mit kleinerem Datensatz

Time:
1-2: 0
2-7: 0
_>7: 1
==> 2/5

Gender:
m: 0
f: 1
==> 2/5

Egal welches man wählt. 
