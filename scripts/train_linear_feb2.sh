torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_snare_1.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_linear_2048_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_snare_2.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_linear_2048_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_snare_3.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_linear_2048_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_open_snare.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_linear_2048_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_noisy_snare.json --data.onset_feature.frame_size 2048 --trainer.default_root_dir logs_linear_2048_train_feb2

torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_snare_1.json --trainer.default_root_dir logs_linear_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_snare_2.json --trainer.default_root_dir logs_linear_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_snare_3.json --trainer.default_root_dir logs_linear_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_open_snare.json --trainer.default_root_dir logs_linear_train_feb2
torchdrum-train-sdss fit -c cfg/onset_mapping_808_linear.yaml --model.preset cfg/presets/808_noisy_snare.json --trainer.default_root_dir logs_linear_train_feb2

torchdrum-test logs_linear_2048_train_feb2/lightning_logs test_logs_linear_2048_train_feb2
torchdrum-test logs_linear_train_feb2/lightning_logs test_logs_linear_train_feb2
