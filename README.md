# Defending Your Voice: Adversarial Attack on Voice Conversion

This is the official implementation of the paper [Defending Your Voice: Adversarial Attack on Voice Conversion](https://arxiv.org/abs/2005.08781).
We use adversarial attack to prevent one's voice from improperly utilized in voice conversion.
The conversion fails when adversarial noise is added to the input utterance.

For the audio samples, please refer to our [demo page](https://yistlin.github.io/attack-vc-demo/).

## Pre-trained Model

The pre-trained [AdaIN-VC](https://arxiv.org/abs/1904.05742) (referred to as Chou's model in the paper) is available [here](https://drive.google.com/file/d/1vBF-4s5u0sro3nwDFWL7VnAV6KViCMp0/view?usp=sharing).
The files are from the official implementation of AdaIN-VC, but the config file was slightly modified.

## Attack

You can perform adversarial attack on AdaIN-VC with `attack.py`.

```bash
python attack.py <model_dir> <vc_tgt> <adv_tgt> <output> [--vc_src source] [--eps epsilon] [--n_iters iterations] [--attack_type type]
```

- **model_dir**: The directory of model files.
- **vc_tgt**: The target utterance to be defended, providing vocal timbre in voice conversion.
- **adv_tgt**: The target used in adversarial attack (**y** in the paper).
- **output**: The output defended utterance.
- **source**: The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).
- **epsilon**: The maximum amplitude of the perturbation.
- **iterations**: The number of iterations for updating the perturbation.
- **type**: The type of adversarial attack to use (end-to-end, embedding, or feedback attack).

## Inference

You can perform voice conversion with `inference.py`.

```bash
python inference.py <model_dir> <source> <target> <output>
```

- **model_dir**: The directory of model files.
- **source**: The source utterance providing linguistic content in voice conversion.
- **target**: The target utterance providing vocal timbre in voice conversion.
- **output**: The output converted utterance.

## Reference

Please cite our paper if you find it useful.

```bib
@article{huang2020defending,
  title={Defending Your Voice: Adversarial Attack on Voice Conversion},
  author={Huang, Chien-yu and Lin, Yist Y and Lee, Hung-yi and Lee, Lin-shan},
  journal={arXiv preprint arXiv:2005.08781},
  year={2020}
}
```
