from myThesis.encoder import nd_on_transformer_encoder
from myThesis.encoder import calculate_IoU_for_encoder
from myThesis.decoder import nd_on_transformer_decoder
from myThesis.decoder import calculate_IoU_for_decoder
from myThesis.lrp import visualise_network
from myThesis.lrp import calculate_network


nd_on_transformer_encoder.main( 
    images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images",
    weights_path="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth",
    output_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/encoder"
)

calculate_IoU_for_encoder.main(
    percentile=0.90, 
    mask_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/rot",
    encoder_out_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/encoder",
    export_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6//rot/encoder", # encoder zu /rot/encoder
    export_mode="global-best",
)


nd_on_transformer_decoder.main(
    images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images",
    weights_path="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth",
    output_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/decoder"
)

calculate_IoU_for_decoder.main_network_dissection_per_query(
    percentile=0.90,
    mask_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/rot",
    decoder_out_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/decoder",
    export_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/decoder", # decoder zu /rot/decoder
    )

calculate_network.main(
    images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images",
		output_root="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6",
		encoder_rot_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/encoder",
		decoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/decoder",
		lrp_out_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp",
		lrp_encoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/encoder",
		lrp_decoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/decoder",
		summary_csv="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/top_features.csv",
    )

visualise_network.main(
    module="encoder",
    top_features="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/top_features.csv",
    encoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/encoder",
    decoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/decoder",
    out="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/visualisations/encoder_graph",
    k=5
    )

visualise_network.main(
    module="decoder",
    top_features="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/top_features.csv",
    encoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/encoder",
    decoder_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/lrp/decoder",
    out="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car/finetune6/rot/visualisations/decoder_graph",
    k=5
    )