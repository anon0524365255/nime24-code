torchdrum fit -c cfg/01_snare1_pop_no_damp.yaml
torchdrum fit -c cfg/02_snare1_pop_damp.yaml
torchdrum fit -c cfg/03_snare1_supra_no_damp.yaml
torchdrum fit -c cfg/04_snare1_supra_damp.yaml
torchdrum fit -c cfg/05_snare2_pop_no_damp.yaml
torchdrum fit -c cfg/06_snare2_pop_damp.yaml
torchdrum fit -c cfg/07_snare2_supra_no_damp.yaml
torchdrum fit -c cfg/08_snare2_supra_damp.yaml
torchdrum fit -c cfg/01_snare1_pop_no_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/02_snare1_pop_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/03_snare1_supra_no_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/04_snare1_supra_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/05_snare2_pop_no_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/06_snare2_pop_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/07_snare2_supra_no_damp.yaml --model.model.hidden_size 512
torchdrum fit -c cfg/08_snare2_supra_damp.yaml --model.model.hidden_size 512
