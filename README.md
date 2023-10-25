## Conditional GIF Synthesis

Generating GIFs based on text descriptions, start frame or end frame.

## Requirements

## Instructions to setup

```bash
conda create -n gifsyn python=3.9
conda activate gifsyn
# Clone repository
git clone https://github.com/andrewfengusa/TextMotionGenerator.git
# Install Pytorch 1.10.0 (**CUDA 11.1**)
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install required pacakges
pip install -r requirements.txt
```

## Training different baseline models
Values for MODEL-CONFIG can be:

```list
videogpt_vq_vae.yaml
vq_diffusion.yaml
```
```bash
 python src/train.py --config-name=train model=MODEL-CONFIG model.do_evaluation=false trainer.devices=[1] trainer.max_epochs=500 logger=tensorboard
 ```
## Dataset
Training and testing on the TumblrGIF (TGIF) dataset. Create folder named "TGIF" in the same directory as the repo. The folder should have the following files/folders:

```list
tgif_vocab.pkl
tgif-v1.0-gulp.tsv
split (folder)
gulp (folder)
```
Split folder should have 3 files - train.txt, valid.txt, test.txt. gulp folder should have 3 sub-folders train, valid and test with gulp data files in each.

## Rendered videos
Validation videos will be rendered after every 10 epochs starting from 0 and will be stored in logs/.
