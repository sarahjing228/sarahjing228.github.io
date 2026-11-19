# Learning Source-Free Domain Adaptation: From Pseudo-Labeling to Statistical Denoising

*Research notes | November 2026*

> A model becoming more certain does not mean it has become more correct. These notes follow that distinction from entropy minimization and pseudo-labeling to an exploratory idea: estimating label reliability while preserving anatomical boundaries.

## 1. The problem: adapting without the original images

Recently, I have been exploring **Source-Free Domain Adaptation (SFDA)**, particularly for medical image segmentation. My initial motivation came from a practical problem: how can we adapt a pretrained deep learning model to a new dataset when the original training data are no longer available?

This is a common challenge in medical imaging. A segmentation model trained on one institution's CT scans may perform differently on images acquired with another scanner, reconstruction protocol, or patient population. Even when the model achieves high accuracy on its original dataset, its performance may deteriorate when encountering unfamiliar anatomical structures or image characteristics.

In conventional unsupervised domain adaptation (UDA), both labeled source data and unlabeled target data are available during training. SFDA considers a more restrictive setting: we have a pretrained source model and unlabeled target-domain images, but no access to the original source training data.

Formally, given a source-trained model \(f_{\theta_s}\) and an unlabeled target dataset \(D_t=\{x_i^t\}_{i=1}^{N}\), the objective is to obtain an adapted model \(f_{\theta_t}\) that performs better on the target domain without accessing the source dataset.

Here, \(x_i^t\) is the \(i\)-th target image, \(N\) is the number of available target images, and \(\theta_s\) and \(\theta_t\) are the network parameters before and after adaptation. The superscript \(t\) in \(x_i^t\) means “target”; it is not a training step. Target labels are unavailable during adaptation, so better target performance is the goal rather than something this setup guarantees.

At first, this seems relatively straightforward: use the pretrained model to generate predictions on the target images, treat those predictions as pseudo-labels, and fine-tune the model.

However, after reading several papers and conducting my own experiments, I began to realize that the central difficulty is not necessarily how to fine-tune a model. It is **how to determine whether the model's own predictions are trustworthy enough to learn from.**

The thread connecting these methods is the quality of their supervision: what signal tells the model how to change, and when can that signal be misleading? I separate the published methods from observations in my own experiments and from the denoising hypothesis I have yet to validate.

## 2. SHOT: Learning Without Source Data

One of the fundamental papers in SFDA is *SHOT: Source Hypothesis Transfer* by Liang et al. (ICML 2020).

SHOT proposes an interesting approach: rather than updating the entire source-trained model, it freezes the classifier and adapts the feature extractor using unlabeled target data.

Consider a neural network decomposed into two components:

\[
f(x)=h(g(x))
\]

Read this equation from the inside out: \(g\) converts an input image \(x\) into a feature representation, then \(h\) converts that representation into class scores. SHOT changes \(g\) while holding \(h\) fixed. Intuitively, it moves target features relative to an existing decision rule rather than learning a new rule from scratch.

The classifier contains decision boundaries learned from the source domain. SHOT assumes that these boundaries can still provide useful supervision, even when the original source images are unavailable.

Therefore, the method updates the target feature representations while keeping the source classifier fixed.

SHOT combines two key ideas:

**Information maximization.** The model is encouraged to make confident predictions on individual target samples while maintaining diversity across predicted classes. This is important because minimizing prediction entropy alone could lead to a degenerate solution where every sample is assigned to the same class.

**Self-supervised pseudo-labeling.** Target features are clustered to obtain pseudo-labels, which provide additional supervision during adaptation.

SHOT++ further develops the framework by transferring labeling information from high-confidence target samples to lower-confidence samples using a semi-supervised learning strategy.

One interesting takeaway is that the pretrained source classifier can serve as a form of prior knowledge. We do not necessarily need the original training samples to preserve every aspect of the source model's behavior.

However, SHOT was originally developed primarily for image classification. Applying similar ideas to medical image segmentation requires additional consideration, especially because anatomical structures have spatial relationships and often highly imbalanced class distributions.


## 3. TENT: Can We Adapt a Model by Minimizing Entropy?

Another method I explored is *TENT: Fully Test-Time Adaptation by Entropy Minimization* (Wang et al., ICLR 2021).

Unlike methods that require a separate target-domain training stage, TENT performs adaptation using incoming test data.

Its main objective is to minimize predictive entropy.

For a probability distribution over \(C\) classes:

\[
H(p)=-\sum_{c=1}^{C}p_c\log p_c
\]

Entropy measures how dispersed the predicted probability distribution is.

Here, \(p_c\) is the predicted probability of class \(c\), the probabilities sum to one, and \(C\) is the number of classes. The sum combines the uncertainty across all possible classes. Since \(\log p_c\leq 0\), the minus sign makes entropy nonnegative; by convention, \(0\log 0=0\). I use natural logarithms, so entropy is measured in nats. Its maximum is \(\log C\) for a uniform prediction, and its minimum is zero for complete certainty.

For example, a binary classifier predicting \([0.5,0.5]\) has high entropy because it is uncertain between the two classes. A prediction of \([0.99,0.01]\) has much lower entropy because it strongly favors one class.

Numerically, \(H([0.5,0.5])\approx0.693\), whereas \(H([0.99,0.01])\approx0.056\). Minimizing entropy rewards a sharper distribution, regardless of whether the favored class is correct.

TENT uses entropy minimization to adjust normalization-related parameters. In its original implementation, Batch Normalization plays a central role: target-batch statistics are used during prediction, and the normalization affine parameters are optimized to reduce entropy.

An attractive aspect of TENT is that it does not require labeled target data or updates to every network parameter.

However, I found that **entropy minimization is not equivalent to accuracy maximization**.

Consider a simplified binary muscle-versus-background example. This illustrates the limitation of the objective; it is not a claim that every entropy-minimization update follows this trajectory. The true class is muscle. Before adaptation, the model might assign:

\[
P(\text{muscle})=0.60
\]

After adaptation, it might assign:

\[
P(\text{muscle})=0.05
\]

The adapted model is now more confident that the voxel is background, even though the prediction is incorrect.

The distribution changes from [0.60, 0.40] to [0.05, 0.95]. Entropy falls from approximately 0.673 to 0.199 nats, but the predicted class changes from correct to incorrect. The entropy objective cannot see that error because it has no ground-truth label.

This distinction is particularly important under severe domain shift, where a model may already have incorrect assumptions about an unfamiliar structure.

There is also an implementation detail relevant to my experiments. My pretrained segmentation network uses Instance Normalization rather than the Batch Normalization configuration assumed by the original TENT method. Adapting InstanceNorm affine parameters through entropy minimization is an interesting experiment, but it should be described as a **TENT-inspired approach**, rather than an exact reproduction of the original algorithm.


## 4. Pseudo-Labeling: Learning From the Model's Own Predictions

Pseudo-labeling is one of the most common mechanisms in SFDA.

The pretrained model first predicts labels for unlabeled target-domain images:

\[
\hat y(v)=\arg\max_c p(c\mid x,v)
\]

where \(v\) is a voxel location and \(p(c\mid x,v)\) is the predicted probability for class \(c\).

The \(\arg\max\) operator returns the **class index**, not its probability. For example, if the probabilities for background, muscle, and bone are [0.10, 0.80, 0.10], the pseudo-label is muscle. The hat on \(\hat y\) reminds us that this is an estimated label, not an annotation.

These predictions are then used as supervision for adapting the model.

The problem is that pseudo-labels may contain errors.

If a voxel is incorrectly classified as background, repeatedly training the model to predict background at that location can reinforce the error. This is a form of confirmation bias in self-training.

A straightforward solution is confidence filtering. We only use pseudo-labels when the maximum predicted probability exceeds a threshold:

\[
m(v)=\mathbb{1}
\left[
\max_c p(c\mid x,v)\geq\tau
\right]
\]

The indicator \(\mathbb{1}[\cdot]\) is 1 when its condition is true and 0 otherwise. Thus, \(m(v)\) acts as an on/off switch for supervision at voxel \(v\), and \(\tau\) controls how selective that switch is.

For example, when \(\tau=0.9\), only predictions with at least 90% maximum class probability contribute to the selected training loss.

A typical masked cross-entropy objective makes the role of the mask explicit:

\[
\mathcal L_{\mathrm{PL}}=-\frac{\sum_v m(v)\log p_\theta(\hat y(v)\mid x,v)}{\max(1,\sum_v m(v))}.
\]

Here, \(p_\theta\) is the model being updated. For this update, the pseudo-labels and masks are treated as fixed targets. Only selected voxels contribute; the denominator averages over their count. If none are selected, the numerator is zero and the denominator is one, so this loss contributes no update. For a selected muscle label, raising its predicted probability from 0.6 to 0.9 reduces its loss from about 0.511 to 0.105. The same reward occurs even when that muscle label is wrong.

However, this approach has a fundamental limitation.

**High confidence does not necessarily indicate correctness.**

A model can be confidently wrong when encountering anatomical abnormalities, unusual intensity distributions, or structures that differ substantially from its source training data.

In my own experiments, confidence-masked pseudo-label training did not reliably improve segmentation compared with the original pretrained model. This suggests that removing low-confidence predictions is insufficient when incorrect labels can also have high confidence.

It also raises an important distinction between three operations:

- **Pseudo-label filtering:** Decide which labels to trust.
- **Pseudo-label weighting:** Reduce the influence of potentially unreliable labels.
- **Pseudo-label refinement:** Attempt to correct the labels themselves.

This distinction has become increasingly important in how I think about my research problem.

## 5. UPL-SFDA: Can Uncertainty Help?

A particularly relevant method for medical image segmentation is *UPL-SFDA: Uncertainty-Aware Pseudo Label Guided Source-Free Domain Adaptation* (Wu et al., IEEE TMI 2023).

UPL-SFDA introduces model perturbations and multiple prediction heads to improve pseudo-label supervision. Its framework includes target-domain growing, reliable pseudo-label generation with twice-forward supervision, and entropy regularization.

The general intuition is that instead of relying on one prediction, we can compare different predictions to obtain more information about their reliability.

Suppose multiple prediction heads produce probability distributions:

\[
p_1(v),p_2(v),\ldots,p_K(v)
\]

Their disagreement can provide information about model uncertainty.

This is related to the distinction between two common forms of uncertainty:

**Aleatoric uncertainty** arises from ambiguity or noise in the observed data, such as unclear boundaries or imaging artifacts.

**Epistemic uncertainty** arises from uncertainty in the model's learned representation and may be approximated using model ensembles, stochastic inference, or parameter perturbations.

To understand disagreement, consider the following general ensemble measure. This is explanatory background, not a statement that UPL-SFDA uses this exact equation. A common ensemble-based proxy for epistemic uncertainty is the difference between the entropy of the averaged prediction and the average entropy of individual predictions:

\[
U_{\mathrm{epi}}(v)
=
H\left(\frac{1}{K}\sum_k p_k(v)\right)
-
\frac{1}{K}\sum_k H(p_k(v))
\]

Here, \(K\) is the number of predictions and each \(p_k(v)\) is a complete class-probability vector. First average the vectors and measure the entropy of that average. Then subtract the average entropy of the individual vectors. The difference measures disagreement: it is the Jensen–Shannon divergence of the predictions with equal weights.

For two binary predictions [0.99, 0.01] and [0.01, 0.99], each head is confident, but their average is [0.5, 0.5]. The score is approximately \(0.693-0.056=0.637\) nats. If both heads predict [0.5, 0.5], the score is zero: they are individually uncertain but do not disagree. If both predict [0.99, 0.01], it is also zero. A low disagreement score therefore does not mean low total uncertainty.

Interpreting this as epistemic uncertainty is strongest when the predictions meaningfully represent uncertainty over models, for example approximate posterior samples. Correlated heads are a practical proxy, not a clean separation of all epistemic and aleatoric effects.

However, an important limitation remains: prediction agreement is not a guarantee of correctness.

If multiple prediction heads share similar biases, they may all make the same incorrect prediction.

This is especially relevant to medical image segmentation, where a model may consistently predict a normal anatomical shape even when the patient's anatomy is significantly deformed.

In my experiments, reproducing UPL-SFDA led to degradation on the held-out source-domain evaluation set. Although this observation does not imply that the published method is ineffective, it demonstrates the importance of evaluating **catastrophic forgetting** when adapting a strong pretrained medical model.


## 6. A Different Perspective: Pseudo-Labeling as a Denoising Problem

Recently, I started thinking about whether pseudo-label errors could be treated similarly to noise in computer graphics.

This idea came from my earlier work on Monte Carlo denoising.

In Monte Carlo rendering, a pixel value is estimated from a finite number of randomly sampled light paths. Because only a limited number of samples are available, the estimate contains stochastic noise.

A denoising algorithm attempts to recover the underlying signal while preserving important image structures.

One influential approach is *Spatiotemporal Variance-Guided Filtering (SVGF)*, which combines temporal accumulation, variance estimates, and spatial filtering to reduce rendering noise while preserving image details.

Although pseudo-label errors and Monte Carlo noise have different origins, I noticed a conceptual similarity:

In both cases, we have an imperfect observation and want to determine which information should be trusted. But the analogy has limits: Monte Carlo estimators can be unbiased under appropriate sampling assumptions, while pseudo-label errors may be systematic and shared across predictions. Averaging can reduce random variation without correcting a shared bias.

For Monte Carlo denoising, we might ask:

*Is the difference between two neighboring pixels caused by random noise, or does it represent a genuine image boundary?*

For segmentation pseudo-labeling, the corresponding question becomes:

*Does a voxel disagree with its neighbors because it has been incorrectly classified, or because it lies near a genuine anatomical boundary?*

This suggests that spatial and feature-based consistency might help identify unreliable pseudo-labels.

For example, given a neighborhood \(\mathcal N(v)\), one could estimate a local label-consistency score:

\[
R(v)=
\frac{
\sum_{u\in\mathcal N(v)}
w(v,u)\mathbb{1}[\hat y(u)=\hat y(v)]
}{
\sum_{u\in\mathcal N(v)}w(v,u)
}
\]

Here, \(u\) is a neighboring voxel and \(\mathcal N(v)\) excludes \(v\) itself to avoid counting self-agreement. The nonnegative weight \(w(v,u)\) measures similarity, potentially using CT intensity, physical distance, or learned features. The numerator sums the weights of neighbors that share the center voxel's pseudo-label; the denominator sums all neighbor weights. Provided the denominator is positive, \(R(v)\) lies between zero and one.

With four equally weighted neighbors, three matching labels give \(R(v)=3/4=0.75\). With weights [0.4, 0.3, 0.2, 0.1], if only the first two neighbors agree, the score is 0.7. Similar neighbors therefore contribute more than dissimilar ones. If all weights are zero, the score is undefined; an implementation should mark that voxel unsupported rather than interpret it as reliable.

This is a **local agreement score**, not a calibrated probability that the label is correct. A whole region can receive a high score while sharing the same wrong label. In 3D CT, physical spacing also matters: one voxel along a thick-slice direction may be much farther away than one voxel in-plane.

However, a simple majority-voting filter is unlikely to be sufficient.

Muscle boundaries, narrow structures, and regions with unusual anatomy may naturally differ from their neighbors. Blind smoothing could remove important information and make the segmentation worse.

Therefore, the more interesting direction is not simply spatial smoothing, but **statistical reliability estimation with boundary preservation**.

This also connects to *Neighborhood Reciprocity Clustering (NRC)*, which exploits neighborhood relationships in learned feature space for source-free adaptation. NRC suggests that similarity between target samples can provide useful adaptation signals without source data.

The difference is that my current interest is more focused on whether statistical relationships between neighboring voxels can help identify incorrect pseudo-labels in 3D medical images.

At this stage, this is a research hypothesis rather than a validated method.


## 7. Connecting SFDA With Federated Learning

Another question I have been exploring is how SFDA can be combined with federated learning.

The two approaches address different problems.

SFDA removes the requirement to access source training images during adaptation. Federated learning allows multiple clients to collaborate without centrally pooling their raw datasets.

In a federated SFDA setting, different medical institutions may each have their own unlabeled target-domain images. They could perform local source-free adaptation and exchange model updates instead of patient images.

A standard federated averaging formulation, following FedAvg, is:

\[
\theta^{r+1}
=
\sum_{k=1}^{K}
\frac{n_k}{\sum_j n_j}\theta_k^{r+1}
\]

Here, \(r\) is the communication round, \(K\) is the number of participating clients, \(n_k\) is the number of local training samples at client \(k\), and \(\theta_k^{r+1}\) is that client's model after local training from the shared starting model. The weights sum to one. For two clients with 100 and 300 samples, the server averages their parameters with weights 0.25 and 0.75. These are dataset-size weights, not measures of pseudo-label quality; a larger client can still provide a harmful update.

However, federated learning is not automatically more computationally efficient, more accurate, or completely privacy-preserving. The quality of local updates and the possibility of information leakage from shared model parameters must still be considered.

In my current experiments, I have also investigated averaging an adapted model with the original pretrained model.

With one target client and a frozen source model, this is more precisely described as source-anchored parameter interpolation than conventional multi-client federated learning.

The motivation is to allow some adaptation to the target domain while limiting excessive deviation from the pretrained parameters.

The interpolation can be written as:

\[
\theta_{\mathrm{mix}}=(1-\alpha)\theta_s+\alpha\theta_{\mathrm{adapt}},\qquad 0\leq\alpha\leq1.
\]

The source model is recovered at \(\alpha=0\); the fully adapted model at \(\alpha=1\). At \(\alpha=0.2\), each parameter moves 20% of the way from its source value toward its adapted value. This assumes matching architectures and aligned parameters, as when adaptation starts from the source checkpoint. Parameter interpolation is not the same as averaging output probabilities, and neural-network nonlinearity means performance need not vary smoothly or improve between the endpoints.

An important question is whether this type of source anchoring can help reduce catastrophic forgetting, especially when target-domain pseudo-labels are unreliable.

## 8. What I Learned From My Experiments

My practical research focuses on adapting a pretrained whole-body musculoskeletal segmentation model to external CT scans, with particular attention to the erector spinae muscle and anatomically challenging scoliosis cases.

The experimental setup includes unlabeled target-domain CT scans for adaptation, manually annotated external cases reserved for evaluation, and held-out source-domain cases for monitoring catastrophic forgetting. The source evaluation set is separate from the data used to update the model. To preserve a strict source-free evaluation protocol, it should not guide checkpoint selection or hyperparameter tuning; using it that way would introduce additional source-domain information. Likewise, target test annotations must remain outside adaptation and model selection.

The observations below are qualitative findings from ongoing experiments, not a controlled benchmark or a general ranking of the methods.

So far, several observations have influenced my understanding of SFDA.

First, a strong pretrained model can be difficult to improve through unsupervised adaptation. When the original model is already reasonably accurate, erroneous pseudo-labels may introduce more harm than useful information.

Second, entropy reduction does not necessarily imply improved segmentation quality. A model can become more confident while its Dice score remains unchanged or decreases.

Third, confidence filtering alone may not solve the problem of erroneous supervision. Some incorrect predictions survive because their confidence is already high.

Fourth, unrestricted adaptation may lead to forgetting. This is particularly problematic if the final model is expected to retain reasonable performance on the original domain.

Finally, I have started to see the importance of separating three questions:

1. How uncertain is the model?
2. How reliable is the generated pseudo-label?
3. Does adaptation using that pseudo-label improve the final segmentation?

These questions are related, but they are not equivalent.

## 9. What would test the denoising hypothesis?

My recent learning has gradually shifted my attention from designing another adaptation strategy toward understanding the quality of the supervision used during adaptation.

Several questions remain open:

- Can local statistical consistency distinguish incorrect pseudo-labels from valid anatomical boundaries?
- Can feature-space similarity provide more reliable information than spatial proximity alone?
- Is it possible to identify confidently incorrect predictions without target-domain ground truth?
- How should unreliable pseudo-labels be filtered, corrected, or weighted during training?
- Can source-anchored adaptation preserve pretrained knowledge while still improving target-domain performance?

A useful next experiment would compare the unchanged source model, confidence filtering, uncertainty weighting, and boundary-aware local consistency under the same adaptation budget. I would report per-structure Dice alongside boundary metrics, source-domain retention, and the fraction of voxels selected for training. Ablations should separate spatial distance, image intensity, and learned features so that any improvement can be attributed to a specific signal.

Recent research, including *Uncertainty-aware Progressive Pseudo-label Denoising (UP2D)*, also explores better pseudo-label refinement through prototypes, uncertainty-aware teacher updates, and selective optimization.

This suggests that pseudo-label reliability is already an active research direction. A useful contribution would need to demonstrate what additional information a statistical or graphics-inspired approach provides beyond existing filtering and refinement methods.


## 10. Conclusion

Studying SFDA has changed how I think about model adaptation.

Initially, I viewed adaptation primarily as an optimization problem: how should we update a pretrained network using unlabeled target images?

Now, I think a more fundamental question is whether the information used to update the network is reliable.

SHOT demonstrates how useful knowledge can be transferred without source data. TENT shows that entropy minimization can provide an adaptation signal at test time. UPL-SFDA introduces uncertainty-aware supervision for medical segmentation. More recent methods investigate how to improve pseudo-label quality through neighborhood relationships, prototypes, and structural consistency.

Yet none of these ideas eliminates the possibility that a model may confidently learn from its own incorrect predictions.

My current interest lies at the intersection of source-free adaptation, uncertainty estimation, statistical reasoning, and computer graphics denoising.

Perhaps the most useful lesson is that **a model becoming more certain does not mean it has become more correct**.

The next step is to investigate whether reliability can be estimated more directly, and whether that reliability can guide adaptation without sacrificing anatomical details.


*These notes reflect my current understanding of the literature and observations from ongoing experiments. The graphics-inspired pseudo-label refinement idea remains exploratory and requires systematic validation.*

## References

1. Liang, J., Hu, D., and Feng, J. **SHOT:** [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://proceedings.mlr.press/v119/liang20a.html). ICML, 2020.
2. Liang, J., et al. **SHOT++:** [Source Data-absent Unsupervised Domain Adaptation through Hypothesis Transfer and Labeling Transfer](https://arxiv.org/abs/2012.07297). IEEE TPAMI.
3. Wang, D., et al. **TENT:** [Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726). ICLR, 2021. [Official implementation](https://github.com/DequanWang/tent).
4. Wu, J., et al. **UPL-SFDA:** [Uncertainty-aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation](https://arxiv.org/abs/2309.10244). IEEE TMI, published online 2023. [Official implementation](https://github.com/HiLab-git/UPL-SFDA).
5. Schied, C., et al. **SVGF:** [Spatiotemporal Variance-Guided Filtering: Real-time Reconstruction for Path Traced Global Illumination](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/). HPG, 2017.
6. Yang, S., et al. **NRC:** [Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation](https://papers.nips.cc/paper/2021/hash/f5deaeeae1538fb6c45901d524ee2f98-Abstract.html). NeurIPS, 2021.
7. McMahan, H. B., et al. **FedAvg:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html). AISTATS, 2017.
8. **UP2D:** [Uncertainty-aware Progressive Pseudo-label Denoising for Source-Free Domain Adaptive Medical Image Segmentation](https://www.sciencedirect.com/science/article/pii/S0925231226000561). Neurocomputing, 2026. [Open preprint](https://arxiv.org/abs/2510.26826).
