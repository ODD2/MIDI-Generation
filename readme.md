# MIDI Generation
## Description
This repository includes the project for the third homework of the course "Deep Learning for Music Analysis and Generation" lectured by Prof. Yang at the National Taiwan University. The main goals of this work is to train a midi generator model on the [pop1k7](https://github.com/YatingMusic/compound-word-transformer) dataset. During inference, the model will randomly generate midi and audio pairs.
## Create Environment 
```bash
pip install -r requirements.txt
```


# Inference
 - Please download the model weights from Google Drive: [Link](https://drive.google.com/file/d/18EH9XdkuELq9IgRPYR8I0OAtQhyrU5t8/view?usp=sharing)
 - Place the checkpoint in the checkpoint folder under a specific name. e.g. checkpoint/model_v1/weight.pkl
 - Inference the model with the following command: 
 ```shell
    python -m main \
    --model_ver=v2 \ 
    --mode=test \
    --ckp_path=checkpoints/model_v1/weight.pkl \
    --num_samples=20 \
    --temperature=3.0 \
    --sample_mode="topk" \
    --topk=3 \
    --out_prefix="temp3d0_topk3"
 ```
 - The command in the example above will locate the generated midi and audios in the folder 'results/model_v1/temp3d0_topk3/'.


    