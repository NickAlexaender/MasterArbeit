Herangehensweise:
myThesis/encoder/nd_on_transformer_decoder.py ausführen
myThesis/encoder/calculate_IoU_for_decoder.py ausführen




Der Code funktioniert, aber muss definitiv noch angepasst werden:

Anpassungen: 
# myThesis/encoder/nd_on_transformer_decoder.py
- siehe Info Encoder

# myThesis/encoder/weights_extraction_transformer_decoder.py
- Teil von nd_on_transformer_edecoder.py
- Funktioniert gut
- Genau durchgehen, ob das was passiert auch robust ist

**Nur Queries berücksichtigen, die aktiv und mit einer Ground-Truth-Instanz aligniert sind**
Wir sollten beides machen:

- Wenn du nur wissen willst, wie gut Farben in Queries gelernt werden, unabhängig von GT-Matching, solltest du auch die inaktiven Queries einbeziehen
-> ob Farben funktional relevant für Prediction sind

- Wenn du dagegen eine „klassische“ mIoU-Bewertung im Sinne von Detection/Segmentation machst, dann bleibst du bei der Regel: nur Queries, die mit GT-Instanzen aligniert sind.
-> ob sie allgemein in der Repräsentation stecken

# myThesis/encoder/calculate_IoU_for_decoder.py
- Ziel ist es hier pro Konzept pro Bild die wichtigsten Feature zu berechnen für die Konzepte.
- myThesis/encoder/iou_core_decoder.py soll dabei die Datei sein, die am alles berechnet. 
-> Auch hier ist wichtig, dass wir den Schwellenwert langfristig Global kalibieren.
**Schwellen global kalibrieren (pro Unit, nicht pro Bild).**



# myThesis/encoder/iou_core_decoder.py
Hier soll langfristig die Hauptberechnung stattfinden. Dieser Teil ist entscheidend und sollte sorgfällig gebaut und nachvollzogen werden.
- Wie entstehen die Heatmaps?
- Nach welchen kriterien wird binarisiert?
- sollten wir alle Level einzeln betrachten oder gemeinsam?


-> Generell sollte alles noch mal genauer nachvollzogen werden und auch getested, dann alles passt