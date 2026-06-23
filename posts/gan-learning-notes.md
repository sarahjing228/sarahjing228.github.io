# How GANs Learn

A generative adversarial network (GAN) learns through a competition between two neural networks. The **generator** creates synthetic samples from random input, while the **discriminator** learns to distinguish generated samples from real data.

## The basic idea

The original minimax objective is:

`min_G max_D E_data[log D(x)] + E_z[log(1 - D(G(z)))]`

The discriminator improves by classifying real and generated samples correctly. The generator improves by producing samples that the discriminator is more likely to accept as real. In practice, the generator often uses the non-saturating loss `-log D(G(z))` to obtain stronger gradients early in training.

## Why training is difficult

- **Imbalance:** if either network learns much faster, the other receives an unhelpful training signal.
- **Mode collapse:** different latent inputs produce too few kinds of outputs.
- **Oscillation:** because both objectives change together, training need not move steadily toward one optimum.
- **Evaluation:** an image can look realistic while failing to represent the diversity or meaning of the real distribution.

## What matters in medical imaging

Visual quality alone is not enough. A synthetic medical image should preserve anatomy, pathology, acquisition characteristics, and quantitative measurements. Evaluation may therefore include expert review, distributional measures, privacy checks, and performance on a carefully chosen downstream clinical task.

## References

- [Goodfellow et al., Generative Adversarial Nets (2014)](https://arxiv.org/abs/1406.2661)
- [Arjovsky, Chintala & Bottou, Wasserstein GAN (2017)](https://arxiv.org/abs/1701.07875)
- [Heusel et al., GANs Trained by a Two Time-Scale Update Rule (2017)](https://arxiv.org/abs/1706.08500)
