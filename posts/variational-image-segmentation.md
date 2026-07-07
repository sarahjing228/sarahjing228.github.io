# Variational Image Segmentation: Energy, Boundaries, and Geometry

This note is about image segmentation from a mathematical-imaging point of view. Instead of beginning with a neural network, variational segmentation begins with an energy function. The segmentation is the object, boundary, or label map that minimizes this energy.

The very compact idea is:

```text
good segmentation = good image explanation + simple / regular geometry
```

In medical imaging, this viewpoint is useful because segmentation is rarely the final product. A mask may later be used to compute organ volume, muscle cross-sectional area, fat fraction, density, surface area, thickness, or longitudinal anatomical change. Therefore, the geometry of the boundary matters.

## 1. Basic notation

Let the image domain be `D`. In 2D, `D` is an image plane. In 3D, `D` is a voxel volume. Let:

```text
I(x) = image intensity at location x
x ∈ D
```

For binary segmentation, the object region can be written as `Ω`, where:

```text
Ω      = foreground / target anatomy
D \ Ω  = background
∂Ω     = boundary of the object
```

Equivalently, segmentation can be represented by an indicator function:

```text
u(x) = 1  if x ∈ Ω
u(x) = 0  if x ∉ Ω
```

For multi-class segmentation, `u(x)` may become a vector of label probabilities:

```text
u_k(x) = probability or indicator for class k at location x
sum_k u_k(x) = 1
```

This is already close to how neural networks represent segmentation maps.

## 2. Energy minimization

A variational segmentation model usually has the form:

```text
minimize over u:

E(u) = Data(I, u) + λ Regularization(u)
```

The data term measures how well the segmentation explains the image. The regularization term controls the geometry of the solution. The parameter `λ` balances the two.

If `λ` is too small:

- the segmentation may follow noise;
- the boundary may become jagged;
- small isolated regions may appear.

If `λ` is too large:

- the segmentation may become oversmoothed;
- thin structures may disappear;
- weak but real anatomical boundaries may be ignored.

This trade-off appears everywhere in medical segmentation: we want masks to follow anatomy, but not every noisy intensity fluctuation.

## 3. Edge-based active contours

Classical active contours, or snakes, evolve a curve so that it moves toward image edges while remaining smooth.

A simplified snake energy is:

```text
E(C) = internal smoothness energy + image edge energy
```

where `C` is the contour. The internal energy penalizes irregular curves. The image energy attracts the contour to strong gradients.

Conceptually:

```text
smooth curve + strong edge alignment = good boundary
```

This works well when object boundaries correspond to visible intensity edges. However, medical images often contain weak edges, partial-volume effects, artifacts, and low contrast between adjacent tissues. For example, muscle and surrounding soft tissue may not always have a sharp boundary in CT or MRI.

This motivates region-based methods.

## 4. Mumford-Shah functional

The Mumford-Shah model is one of the central variational models for image segmentation. It assumes that an image can be approximated by smooth regions separated by boundaries.

In words:

```text
observed image = smooth approximation inside regions + discontinuities at boundaries
```

A simplified form is:

```text
E(u, K) =
  ∫_{D \ K} |I(x) - u(x)|^2 dx
  + α ∫_{D \ K} |∇u(x)|^2 dx
  + β Length(K)
```

where:

- `I` is the observed image;
- `u` is a piecewise smooth approximation of the image;
- `K` is the boundary or discontinuity set;
- the first term fits the image;
- the second term encourages smoothness inside regions;
- the third term penalizes overly long boundaries.

The full Mumford-Shah problem is difficult to optimize because both the smooth approximation `u` and the unknown boundary set `K` must be estimated. But it is conceptually important because it connects segmentation with piecewise smooth image modeling.

## 5. Chan-Vese active contour

The Chan-Vese model is a widely used simplification of Mumford-Shah. It is region-based rather than edge-based. It does not require the boundary to coincide with a strong gradient.

For two regions, the model assumes:

```text
inside the object:  intensity ≈ c1
outside the object: intensity ≈ c2
```

The energy can be written conceptually as:

```text
E(C, c1, c2) =
  μ Length(C)
  + ν Area(inside C)
  + λ1 ∫inside(C)  |I(x) - c1|^2 dx
  + λ2 ∫outside(C) |I(x) - c2|^2 dx
```

where:

- `C` is the contour;
- `c1` is the average intensity inside the contour;
- `c2` is the average intensity outside the contour;
- `Length(C)` regularizes the boundary;
- the data terms measure how well the two-region model fits the image.

The model tries to find a contour such that pixels inside the contour are statistically similar to `c1`, while pixels outside are statistically similar to `c2`.

This is useful for medical images where boundaries are not always sharp. For example, a structure may be better described by regional intensity statistics than by a single strong edge.

## 6. Level-set representation

Instead of storing the contour directly, level-set methods represent it implicitly using a function `φ(x)`.

```text
inside object:  φ(x) > 0
outside object: φ(x) < 0
boundary:       φ(x) = 0
```

The boundary is the zero level set:

```text
∂Ω = { x | φ(x) = 0 }
```

This representation is powerful because the contour can change topology. It can split, merge, shrink, or expand naturally during evolution.

For 3D medical images, this becomes especially useful:

```text
2D contour  -> curve
3D boundary -> surface
```

The level-set function embeds the surface inside a volumetric scalar field. This connects segmentation directly to geometry and surface evolution.

## 7. Gradient descent intuition

Many variational models are optimized by gradient descent. The idea is to evolve the segmentation in the direction that decreases the energy:

```text
u_{t+1} = u_t - η · gradient_of_E(u_t)
```

For level sets, the function `φ` evolves over artificial time:

```text
∂φ / ∂t = force terms
```

The force terms may include:

- image forces, which move the boundary toward better data agreement;
- curvature forces, which smooth the boundary;
- balloon forces, which expand or shrink the contour;
- region forces, which compare inside and outside intensity models.

This is helpful conceptually: segmentation is not just classification at each voxel. It can also be understood as boundary motion driven by image statistics and geometric regularity.

## 8. Total variation regularization

Total variation, often written as `TV(u)`, is another important mathematical idea:

```text
TV(u) = ∫_D |∇u(x)| dx
```

For binary segmentation, total variation is closely related to boundary length in 2D or surface area in 3D. Penalizing total variation encourages simpler, cleaner boundaries.

A typical segmentation energy may look like:

```text
E(u) = ∫_D data_cost(x, u(x)) dx + λ TV(u)
```

This is a nice formulation because:

- the data cost decides which label fits each pixel or voxel;
- the TV term discourages noisy label changes;
- the result often has cleaner regions and fewer isolated artifacts.

In medical imaging, total variation-like ideas are related to why smoothness constraints and post-processing can improve anatomical plausibility.

## 9. Graph cuts

Graph-cut methods formulate segmentation as an energy minimization problem on a graph.

Each pixel or voxel is a node. Neighboring nodes are connected by edges. The model usually contains:

```text
unary term:   cost of assigning a label to a voxel
pairwise term: cost of assigning different labels to neighboring voxels
```

A typical discrete energy is:

```text
E(L) = Σ_i D_i(L_i) + Σ_(i,j) V_ij(L_i, L_j)
```

where:

- `L_i` is the label assigned to voxel `i`;
- `D_i` is the data cost;
- `V_ij` is the smoothness cost between neighboring voxels.

For some binary segmentation problems, graph cuts can find a global optimum efficiently. This is one reason they became important in interactive segmentation and medical image analysis.

## 10. Relationship to deep learning

Modern CNN and transformer-based segmentation models usually learn the mapping:

```text
image -> segmentation mask
```

However, many losses and post-processing methods still echo variational ideas.

Examples:

- Cross-entropy loss corresponds to voxel-wise label likelihood.
- Dice loss encourages region overlap.
- Boundary loss emphasizes surface alignment.
- Smoothness losses act like regularizers.
- Shape priors encode anatomical plausibility.
- Conditional random fields and graph-based post-processing impose spatial consistency.

So even when the model is a neural network, the mathematical questions remain:

```text
Does the mask fit the image?
Is the boundary anatomically plausible?
Is the surface stable enough for measurement?
Does the segmentation generalize across scanners and protocols?
```

## 11. Why this matters for medical segmentation evaluation

For medical segmentation, evaluation should not only ask whether voxels overlap. It should also ask whether the segmentation is useful for downstream analysis.

Common region metrics:

- Dice similarity coefficient;
- Jaccard index / IoU;
- sensitivity and specificity;
- precision and recall.

Boundary and surface metrics:

- Hausdorff distance;
- 95th percentile Hausdorff distance;
- average symmetric surface distance;
- surface Dice.

Quantitative biomarker metrics:

- volume error;
- surface area error;
- density or intensity measurement error;
- longitudinal consistency;
- anatomical plausibility.

This is especially important when segmentation is used for CT-based musculoskeletal analysis. A small surface shift may have a limited effect on Dice but still affect volume, muscle density, or fat-infiltration estimates.

## 12. Personal takeaway

Variational segmentation gives a language for thinking about masks as geometric objects. It makes explicit the balance between image evidence and anatomical regularity.

For me, the important lesson is:

```text
segmentation is not only pixel classification;
it is also geometric modeling.
```

This perspective connects medical image analysis with optimization, differential geometry, surface reconstruction, and computer graphics.

## References

- Kass, M., Witkin, A., & Terzopoulos, D. (1988). Snakes: Active contour models. *International Journal of Computer Vision*.
- Mumford, D., & Shah, J. (1989). Optimal approximations by piecewise smooth functions and associated variational problems. *Communications on Pure and Applied Mathematics*.
- Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations. *Journal of Computational Physics*.
- Chan, T. F., & Vese, L. A. (2001). Active contours without edges. *IEEE Transactions on Image Processing*.
- Boykov, Y., & Jolly, M.-P. (2001). Interactive graph cuts for optimal boundary and region segmentation of objects in N-D images. *ICCV*.
- Chambolle, A. (2004). An algorithm for total variation minimization and applications. *Journal of Mathematical Imaging and Vision*.
- Boykov, Y., & Funka-Lea, G. (2006). Graph cuts and efficient N-D image segmentation. *International Journal of Computer Vision*.
