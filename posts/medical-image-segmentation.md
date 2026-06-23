# Segmentation: From Voxels to Meshes

Image segmentation assigns a meaningful label to each pixel in a 2D image or each voxel in a 3D volume. In medical imaging, those labels may represent organs, muscles, tumors, bones, vessels, or other regions needed for measurement and clinical analysis.

## Classification, detection, and segmentation

**Classification** gives one label to an entire image or volume: for example, whether a scan contains a finding. **Detection** locates an object, usually with a point or bounding box. **Segmentation** estimates the complete spatial region occupied by that object. The output is therefore a mask with the same spatial coordinate system as the image.

## Different kinds of segmentation

- **Binary segmentation:** foreground versus background.
- **Multi-class semantic segmentation:** every pixel or voxel receives one of several anatomical labels.
- **Multi-label segmentation:** a location may belong to more than one label when categories overlap conceptually.
- **Instance segmentation:** separates individual objects of the same class, such as distinct lesions.
- **Panoptic segmentation:** combines semantic classes with individual object instances.

Segmentation can also be described by its supervision: fully supervised masks, weak labels such as boxes or points, semi-supervised learning with labeled and unlabeled images, or unsupervised/self-supervised representation learning.

## How a segmentation model works

A typical model first encodes local image patterns into features and then decodes those features back to the original spatial resolution. Skip connections can preserve fine boundaries while deeper features provide context. During training, the predicted mask is compared with an expert annotation using losses such as cross-entropy, Dice loss, focal loss, or combinations of region and boundary terms.

Preprocessing is part of the method: voxel spacing, intensity normalization, cropping, interpolation, patch sampling, and augmentation can all change the task. In 3D CT, it is important to preserve physical units and patient-level train/test separation.

## From a mask to a 3D mesh

A segmentation mask is a regular grid of labeled voxels. To create a surface mesh:

1. Select the target label and define an isovalue between background and foreground.
2. Run a surface-extraction method such as **Marching Cubes** through the voxel grid.
3. Interpolate where the surface crosses each grid cell and connect those points into triangles.
4. Convert voxel coordinates into physical coordinates using image spacing, origin, and orientation.
5. Optionally remove small components, smooth the surface, repair topology, or reduce the triangle count.

Every cleanup step can alter geometry. If the mesh will be used for volume, thickness, or shape analysis, the measurement error introduced by smoothing and decimation should be checked rather than assumed harmless.

## How should medical segmentation be evaluated?

No single score is enough. **Dice similarity coefficient** measures region overlap, but a high Dice score can hide a locally poor boundary. Surface Dice and Hausdorff distance describe boundary agreement. Volume error matters when segmentation supports quantitative biomarkers. Sensitivity and precision show whether the model systematically misses or over-segments a structure.

The most important evaluation depends on the clinical or research use:

- Report results per structure and per patient, not only a global average.
- Inspect difficult and failed cases visually.
- Use patient-level splits and test data from relevant scanners or institutions.
- Measure boundary quality for small or thin anatomy.
- Evaluate downstream quantities such as volume, attenuation, muscle area, or shape.
- Compare model disagreement with inter-observer variability when multiple expert labels are available.

For medical segmentation, the key question is not only “does the mask overlap?” but “is it accurate enough for the decision or measurement it supports?”

## References

- [Ronneberger, Fischer & Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)
- [Lorensen & Cline, Marching Cubes: A High Resolution 3D Surface Construction Algorithm (1987)](https://doi.org/10.1145/37402.37422)
- [Maier-Hein et al., Why rankings of biomedical image analysis competitions should be interpreted with care (2018)](https://doi.org/10.1038/s41467-018-07619-7)
- [Reinke et al., Understanding metric-related pitfalls in image analysis validation (2024)](https://doi.org/10.1038/s41592-023-02150-0)
