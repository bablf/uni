"""
Autor: Florian Babl
Thema: Probing Classifiers for LM in a Chess setting
"""


""" GPT-2 MODELLE """
# Auf welche Modelle habe ich zugriff?
#   1. Auf gpt2 mit UCI & PGN notation finetunen (Muss ich selber trainieren/programmieren)
#       habe PGN Model from Noever aka Chess Transformer
#       Muss UCI Model implementieren
#   2. Auf gpt2 mit converted PGN finetunen
#   3. pretrained gpt2 (kann einfach aus Huggingface importiert werden) # PGN to UCI exists. From Blindfolded Paper
# Todo: Alle 4 Modelle implementieren und aufrufen können.
# Todo selbst implementiertes finetuning trainieren lassen.
# Todo: brauche Daten um finetuning zu machen.



""" PROBING """
# Todo: Wie kann Hidden State extrahiert werden
# TODO: Probing Classifier erstellen, und trainieren.
# 1. Neuronales Netz ausdenken für alle Probing Classifier
# 2. Few-short results implementieren für gpt-2 # Todo: in few-shot einlesen. in bezug auf fine-tuning (mit oder ohne)
# Todo: Datensatz für probing erstellen (Muss der der gleiche für beide finetuned GPT2 modelle sein? Wäre gut.)
# Datensatz erstellen, der Überprüft werden soll.
# Aufbau von Datensatz: String von PGN UND UCI notation --> Schachfeld.
# beides, damit man sowohl PGN als auch UCI nutzen könnte.





if __name__ == '__main__':
    pass


