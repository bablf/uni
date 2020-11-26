# Übung 2

## 2 - 1
a) siehe Blockblatt **correct**

b)  
-----------------rot gelb blau

**wrong - richtiges ranking, aber euclidean distance nicht berechent**
1. d(q,a) =  0  + 4 + 3  = 7
2. d(q,c) =  1  + 4 + 3 = 8
3. d(q,d) =  0 +  6  +6 = 12
4. d(q,b) =  7  + 7 + 0  = 14

c)
**correct**
1. Define a candidate set C that includes only images with 1 red quare and 16 colored quares. Then compare only these with q.
2. Think of / create a hierachical index structure. For example: All Square images and all non quare images get seperated. Then the squared images get split up in images with 1-2 red fields and more than 2 red fields. Then we only need to compute the distance for all images in one "box".

d) Distance function only pays atttention to the colors but not shape and location of the colors

## 2 - 2
1. Product (distributive): We multiply the precomputed result with the new result to get the updated precomputed results
2. Mean (algebraic): **not distributive**

  sum(oldD) + sum(newD) / count(oldD) + count(newD) = newMean

  Das heißt, dass wir 4 bzw 2 Werte brauchen.

3. Variance (algebraic):
  0. n = count(oldD)+count(newD)
  1. neuen Erww. berechenen =

    sum(oldD)+sum(newD) / n

  2. Varianz berechnen:

    Var= sum(Erww.-xi)²/n = (sum(oldD)+sum(newD))² − n⋅Erww²

  --> Wie beim mean 4 bzw. 2 stück

4. Median (holistic)

## 2 - 3
a)
1. equi-width = 9-1/3 = 2.66
Bins: 1-3.66, 3.66-6.33, 6.33-9
2. equi-heigth = 1-4, 5 , 6-9

b)
Im equi-height histogram wäre letzte bin höher.
Das equi-width histogram würde komplett anders aussehen. width = 9,333. Es würde 3 Bins geben, aber es wären alle Einträge 1-9 darin gelistet. Im mittleren Balken wäre nichts. Und im letzten wäre 29
