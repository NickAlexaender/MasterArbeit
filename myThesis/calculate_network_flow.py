#. Wir haben 2 verschiedene Modelle (Autos und Butterflys)

# Wir haben 6 verschiedene Trainingszustände pro Modell 

# Wir testen auf 5 verschiedene Konzepte (Rot, Grün, Blau, Schwarz, Weiß)

# Auf 100 Bilder pro Modell
"""
models = ["cars", "butterflies"]
train_states = ["0", "1", "2", "3", "4", "5"]
concepts = ["red", "green", "blue", "black", "white"]
num_images = 100
"""
# Die Idee ist, dass jedes Modell einen Ort hat in dem alle Modelle liegen: myThesis/output/car_parts_finetune
# Und die Modelle dann auch immer die gleichen Namen haben: model_0000499.pth, model_0000999.pth, model_0001499.pth, model_0001999.pth, model_0002499.pth, model_0002999.pth, model_final.pth
# Zudem gibt es pro Modell einen Ordner mit den Bildern: myThesis/data/cars/images_val



# for model in models:
# for train_state in train_states:
#    first code mit Inhalt -> nd_on_transformer
#    for concept in concepts:
#        sencond code mit Inhalt -> calculate_IoU
#        lrp on all pro 5 Features pro Layer 1-6, 1-3
#        design output


from myThesis.encoder import nd_on_transformer_encoder
from myThesis.encoder import calculate_IoU_for_encoder
from myThesis.decoder import nd_on_transformer_decoder
from myThesis.decoder import calculate_IoU_for_decoder
from myThesis.lrp import visualise_network
from myThesis.lrp import calculate_network


images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images"
weights_path="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth"
model = "car"
train_state="finetune6"
concept="rot"
basic_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/"


'''
nd_on_transformer_encoder.main(
    images_dir=images_dir,
    weights_path=weights_path,
    output_dir=f"{basic_root}output/{model}/{train_state}/encoder"
)



# Wir brauchen zugriff auf 
# -> den Ort wo die Bilder zu finden sind
# -> Das konkrete Modell + Trainingsstate, dass verwendet wird
# -> Den Ort wo die Daten gespeichert werden sollen -> myThesis/output/car/fintune6/encoder...

calculate_IoU_for_encoder.main(
    percentile=0.90, 
    mask_dir=f"{basic_root}image/{concept}",
    encoder_out_dir=f"{basic_root}output/{model}/{train_state}/encoder",
    export_root=f"{basic_root}output/{model}/{train_state}/{concept}/encoder", # encoder zu /{concept}/encoder
    export_mode="global-best",
)
# Wir brauchen zugriff auf 
# -> den Ort wo die Masken zu finden sind
# -> den Ort wo die benötigte Csv Datei zu finden ist.
# -> Den Ort wo die Daten gespeichert werden sollen -> .../encoder...



nd_on_transformer_decoder.main(
    images_dir=images_dir,
    weights_path=weights_path,
    output_dir=f"{basic_root}output/{model}/{train_state}/decoder"
)

calculate_IoU_for_decoder.main_network_dissection_per_query(
    percentile=0.90,
    mask_dir=f"{basic_root}image/{concept}",
    decoder_out_dir=f"{basic_root}output/{model}/{train_state}/decoder",
    export_root=f"{basic_root}output/{model}/{train_state}/{concept}/decoder", # decoder zu /{concept}/decoder
    )
'''
calculate_network.main(
        images_dir=images_dir,
        output_root=f"{basic_root}output/{model}/{train_state}",
        encoder_rot_dir=f"{basic_root}output/{model}/{train_state}/{concept}/encoder",
        decoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/decoder",
        lrp_out_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp",
        lrp_encoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/encoder",
        lrp_decoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/decoder",
        summary_csv=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/top_features.csv",
        weights_path=weights_path,
    )

visualise_network.main(
    module="encoder",
    top_features=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{basic_root}output/{model}/{train_state}/{concept}/visualisations/encoder_graph",
    k=5
    )

visualise_network.main(
    module="decoder",
    top_features=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{basic_root}output/{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{basic_root}output/{model}/{train_state}/{concept}/visualisations/decoder_graph",
    k=5
    )

### Nächste Schritte:

# TODO: Freitag: LRP überarbeiten, sodass die Aussage der Werte die richtige ist! (Aufaddieren der LRP-Werte -> Gesamteinfluss muss 1 sein)
# - vereinfachen -> ganzen Schnickschnack entfernen
# - aufteilen auf mehrere Dateien
# - genau definieren was ich wissen will und calculieren.
# TODO: Samstag: Einmal für 100 Bilder durchlaufen lassen - Zeit aufnehmen

# TODO: Mittwoch: Zweites Modell (Butterflys) integrieren
# TODO: Sonntag: Calculate IoU für Decoder und Encoder genauer untersuchen (-> mit KI durchleuchten)

# TODO: Mittwoch - Frei
# TODO: Donnerstag - Frei

# TODO: Donnerstag: Umbauen, sodass es für verschiedene Modelle funktioniert
# TODO: Freitag: Code in myThesis/calculate_network_flow.py so umbauen, dass alle Modelle, Trainingszustände und Konzepte durchlaufen können.

# TODO. Mittwoch: Jeden Code einzeln durchgehen - logic checken und kommentieren.

