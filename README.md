# DynaSketch: Abstracting Coherent Sketches for Dynamic Image Sequences

This is the official implementation of DynaSketch. A method for converting dynamic image sequences to coherent sketches. <br>

<br>
<br>

![](repo_images/overall.png?raw=true)

    
<br>

## Installation

### Installation via pip [Recommended]
Note that it is recommended to use the provided docker image, as we rely on diffvg which has specific requirements and does not compile smoothly on every environment.
1.  Clone the repo:
```bash
git clone https://github.com/wshjtuk/DynaSketch.git
cd DynaSketch
```
2. Create a new environment and install the libraries:
```bash
python3.7 -m venv dynasketch
source clipsketch/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
```
3. Install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```
4. Install InSpyReNet:
```bash
pip install transparent-background
```
You may also need to download some CKPT from the repo.
5. Install Co-Tracker:

By following the instructions in the [Co-Tracker](https://github.com/facebookresearch/co-tracker) repo, install the library.
You may also need to download some CKPT from the repo.
<br>

## Run Demo

<!-- #### Run a model on your own image -->

The input videos to be drawn should be located under "target_videos".
You can edit the file "run.sh" to specify the target file by editing as:
```bash
folder_list=("<file_name>")
```
and the level of abstraction by editing as:
```bash
stroke_num=(<num_strokes>)
```
To sketch your own dynamic image sequences, run:
```bash
bash run.sh
```
The resulting sketches will be saved to the "output_sketches" folder, in SVG format.

Optional arguments:
* ```--num_strokes``` Defines the number of strokes used to create the sketch, which determines the level of abstraction. The default value is set to 16, but for different images, different numbers might produce better results. 
* ```--mask_object``` It is recommended to use images without a background, however, if your image contains a background, you can mask it out by using this flag with "1" as an argument.
* ```--fix_scale``` If your image is not squared, it might be cut off, it is recommended to use this flag with 1 as input to automatically fix the scale without cutting the image.
* ```--num_sketches``` As stated in the paper, by default there will be three parallel running scripts to synthesize three sketches and automatically choose the best one. However, for some environments (for example when running on CPU) this might be slow, so you can specify --num_sketches 1 instead.
* ```-cpu``` If you want to run the code on the cpu (not recommended as it might be very slow).
* ```--frame_cut``` If you want to cut the video at special intervals (fps) for abstraction, you can change this parameter.


## Related Work
[CLIPDraw](https://arxiv.org/abs/2106.14843): Exploring Text-to-Drawing Synthesis through Language-Image Encoders, 2021 (Kevin Frans, L.B. Soros, Olaf Witkowski)

[CoTracker](https://arxiv.org/abs/2307.07635): Co-Tracker: CoTracker: It is Better to Track Together, 2023 (Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht)

[InSpyReNet](https://arxiv.org/abs/2209.09475): InSpyReNet: Revisiting Image Pyramid Structure for High Resolution Salient Object Detection, 2022 (Taehun Kim, Kunhee Kim, Joonyeong Lee, Dongmin Cha, Jiho Lee, Daijin Kim)

[CLIPasso](https://arxiv.org/abs/2202.05822): CLIPasso: Semantically-Aware Object Sketching, 2022 (Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amir Zamir, Ariel Shamir)

[Diffvg](https://github.com/BachiLi/diffvg): Differentiable vector graphics rasterization for editing and learning, ACM Transactions on Graphics 2020 (Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, Jonathan Ragan-Kelley)



## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
# DynaSketch-2024
