from myThesis.lrp import visualise_network

images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/1images"
weights_path="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0000399.pth"
model = "car"
train_state="finetune1"
concept="grau"



# Lokaler Pfad für Input-Daten (Bilder, Masken, etc.)
local_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/"
# Externer Speicher für Output-Daten (Ergebnisse)
input_root="/Volumes/Untitled/Master-Arbeit_Ergebnisse/output/"
output_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/"

visualise_network.main(
    module="encoder",
    top_features=f"{input_root}{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{output_root}/{train_state}/visualisations/encoder_graph",
    coverage=0.10
    )

visualise_network.main(
    module="decoder",
    top_features=f"{input_root}{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{output_root}/{train_state}/visualisations/decoder_graph",
    coverage=0.10
    )

train_state="finetune2"

visualise_network.main(
    module="encoder",
    top_features=f"{input_root}{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{output_root}/{train_state}/visualisations/encoder_graph",
    coverage=0.10
    )

visualise_network.main(
    module="decoder",
    top_features=f"{input_root}{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{output_root}/{train_state}/visualisations/decoder_graph",
    coverage=0.10
    )

train_state="finetune3"

visualise_network.main(
    module="encoder",
    top_features=f"{input_root}{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{output_root}/{train_state}/visualisations/encoder_graph",
    coverage=0.10
    )

visualise_network.main(
    module="decoder",
    top_features=f"{input_root}{model}/{train_state}/{concept}/lrp/top_features.csv",
    encoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/encoder",
    decoder_dir=f"{input_root}{model}/{train_state}/{concept}/lrp/decoder",
    out=f"{output_root}/{train_state}/visualisations/decoder_graph",
    coverage=0.10
    )