# torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_snare_1.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_mlp_lrg_2048_train_feb2
# torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_snare_2.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_mlp_lrg_2048_train_feb2
# torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_snare_3.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_mlp_lrg_2048_train_feb2
# torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_open_snare.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_mlp_lrg_2048_train_feb2
# torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_noisy_snare.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_mlp_lrg_2048_train_feb2

torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_snare_1.json --trainer.default_root_dir logs_mlp_lrg_train_feb2_alt
torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_snare_2.json --trainer.default_root_dir logs_mlp_lrg_train_feb2_alt
torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_snare_3.json --trainer.default_root_dir logs_mlp_lrg_train_feb2_alt
torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_open_snare.json --trainer.default_root_dir logs_mlp_lrg_train_feb2_alt
torchdrum-train-sdss fit -c cfg/onset_mapping_808_lrg.yaml --model.preset cfg/presets/808_noisy_snare.json --trainer.default_root_dir logs_mlp_lrg_train_feb2_alt
