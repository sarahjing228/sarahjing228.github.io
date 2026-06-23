# Geometry & Linear Algebra for Imaging

Medical imaging and computer graphics both describe objects in space. Linear algebra supplies the language for coordinates and transformations; geometry asks which spatial properties are preserved and how shapes should be represented.

## Vectors and coordinate systems

A point can be written as a vector of coordinates, but those numbers only make sense relative to a coordinate system. Medical volumes have voxel indices as well as physical patient coordinates. Image origin, voxel spacing, and orientation define the transformation between them.

This distinction matters when aligning scans, exporting a mesh, measuring a distance, or overlaying a prediction. Two arrays with the same shape do not necessarily describe the same physical region.

## Transformations

A matrix can represent rotation, scaling, reflection, or shear. Translation is included by using **homogeneous coordinates**, which express an affine transformation as one matrix multiplication.

`p_world = T p_image`

Rigid transformations preserve distances and angles. Affine transformations preserve straight lines and parallelism but may change lengths and angles. Deformable transformations allow local changes and are useful for anatomical registration, but they require constraints and careful interpretation.

## Surfaces, normals, and meshes

A triangular mesh approximates a continuous surface using vertices, edges, and faces. Face and vertex normals describe local orientation and are used for lighting, rendering, and geometric measurements. Mesh quality also depends on topology: holes, self-intersections, non-manifold edges, and inconsistent triangle orientation can cause analysis or rendering failures.

## Connection to ray tracing

Ray tracing models a line from a camera through a scene and asks where that ray intersects a surface. The calculation brings together coordinate transformations, vector normalization, dot products, surface normals, and acceleration structures. The same geometric foundations appear in medical volume rendering, projection imaging, and simulation.

## What I want to study next

- Eigenvalues and singular value decomposition for principal directions and dimensionality reduction
- Barycentric coordinates and interpolation on triangles
- Differential geometry for curvature and local surface shape
- Quaternions and stable 3D rotations
- Numerical precision when transformations are chained

## References

- [Gilbert Strang, Introduction to Linear Algebra resources, MIT](https://math.mit.edu/~gs/linearalgebra/)
- [MIT OpenCourseWare, Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Möller & Trumbore, Fast, Minimum Storage Ray-Triangle Intersection (1997)](https://doi.org/10.1080/10867651.1997.10487468)
- [Lorensen & Cline, Marching Cubes (1987)](https://doi.org/10.1145/37402.37422)
