Herangehensweise:
myThesis/encoder/nd_on_transformer_encoder.py ausführen
myThesis/encoder/calculate_IoU.py ausführen




Der Code funktioniert, aber muss definitiv noch angepasst werden:

Anpassungen: 
# myThesis/encoder/nd_on_transformer_encoder.py
- Regelt genau, welche Daten eingehen und welche nicht
- Ordner mit allen Bildern, die überprüft werden sollen, sollte hier angefügt werden. (Wichtig vielleicht direkt aus Datensatz ziehen, wichtig ist Nachvollziebarkeit)

# myThesis/encoder/weights_extraction_transformer_encoder.py
- Teil von nd_on_transformer_encoder.py
- Funktioniert gut
- Nur Logik wie man Bilder-Daten speichert müssen wir vielleicht noch verändern

# myThesis/encoder/calculate_IoU.py
- Ziel ist es hier pro Konzept pro Bild die wichtigsten Feature zu berechnen für die Konzepte.
- myThesis/encoder/iou_core.py soll dabei die Datei sein, die am alles berechnet. 
(Zu beachten ist hier gerade noch, dass )
**Schwellen global kalibrieren (pro Unit, nicht pro Bild).**
Per-Bild-Perzentile (z. B. 95%) machen jede Heatmap künstlich gleich „scharf“ und verzerren IoU. Kalibriere je (layer_idx, feature_idx, [level_idx]) einen globalen Schwellenwert aus einer Bildmenge (z. B. fester Aktivierungs-Quantilwert der Unit über den Korpus) und fixiere ihn dann.
Output sollte sein: Mit den und den Werten als Vorraussetzung sind die und die Feature in Layer X mit dem Konzept Y wichtig.
-> 99–99.5 % Percentil für Hauptmetrik, und 90–95 % als Robustheits-Check
(Wir wollen nur die Spizen abfangen. Dadurch das wir das über viele Bilder machen, kann dann ein Bild aber auch besonders viele Aktivierungen haben, wenn es zum Beispiel wirklich ein rotes Auto ist und wir Rot versuchen zu messen)


# myThesis/encoder/iou_core.py
Hier soll langfristig die Hauptberechnung stattfinden. Dieser Teil ist entscheidend und sollte sorgfällig gebaut und nachvollzogen werden.
- Wie entstehen die Heatmaps?
- Nach welchen kriterien wird binarisiert?
- sollten wir alle Level einzeln betrachten oder gemeinsam?