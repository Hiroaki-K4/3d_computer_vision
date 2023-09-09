# 3D reconstruction of a plane surface
Shows the procedure for calculating the position of a plane and the position and orientation of a camera from a homography matrix (projective transformation matrix). Since multiple solutions are obtained at this time, we describe the procedure for selecting the correct solution from them.

<br></br>

## Self-calibration by plane
In order to perform [planar triangulation](https://medium.com/@hirok4/implementation-of-planar-triangulation-c66ef654c7fa), we need the camera matrices $P,P\prime$ of the two cameras and the equation of the plane. We consider a self-calibration that computes this from only the homography matrix $H$ between the two images. The homography matrix is indefinite by a constant factor and has 8 degrees of freedom. The equation of the plane is determined by 3 parameters. Therefore, to self-calibrate, the camera matrices $P,P\prime$ must be expressed in terms of 5 parameters. The unknowns are the two focal lengths $f,f\prime$ and the relative translations $t$ (2 degrees of freedom) and rotations $R$ (3 degrees of freedom) of the camera, for a total of 7 degrees of freedom. To reduce this to 5 DoFs, we assume that the focal lengths $f,f\prime$ are known.

The plane is represented by the following form.

$$
n_1X+n_2Y+n_3Z=h \tag{1}
$$

Since the whole represents the same plane no matter how many times it is multiplied, it is multiplied by a constant and normalized so that $n=(n_1, n_2, n_3)^\intercal$ is the unit vector. The sign of $h$ is positive in the direction of $n$ and negative in the opposite direction.
Consider how to compute the camera motion parameters $(R, t)$ and planar parameters $(n, h)$ from the homography matrix $H$. In calculations from images only, the absolute scale of the scene is undefined. Therefore, we assume that the translation of the camera is nonzero and compute a solution whose length is 1.

<img src='images/plane.png' width='300'>

<br></br>

## Computation of planar and motion parameters by homography decomposition
When the focal lengths $f,f\prime$ of the two cameras are known, the motion and planar parameters of the cameras are calculated from the homography matrix $H$ as follows However, it is assumed that

- Camera translation is not zero (we are looking at the plane from different points)
- The plane does not pass through either camera's viewpoint (it is seen as a plane)
- The camera positions are on the same side of the plane (looking at the same side of the plane)
- Distance to the plane is $h>0$ (unit normal vector $n$ of the plane is taken in the direction away from the camera)

In this case, four solutions are obtained as follows.

### 1. Convert homography matrix $H$

$$
\tilde{H}=\begin{pmatrix}
f_0 & 0 & 0 \\
0 & f_0 & 0 \\
0 & 0 & f\prime \\
\end{pmatrix}H
\begin{pmatrix}
1/f_0 & 0 & 0 \\
0 & 1/f_0 & 0 \\
0 & 0 & 1/f\prime \\
\end{pmatrix} \tag{2}
$$

### 2. Normalize matrix $H$ to determinant 1

$$
\tilde{H} \leftarrow \frac{\tilde{H}}{\sqrt[3]{| \tilde{H} |}} \tag{3}
$$

### 3. Singular value decomposition of $H$($U$ and $V$ are orthogonal matrix)

$$
\tilde{H}=U
\begin{pmatrix}
\sigma_1 & 0 & 0 \\
0 & \sigma_2 & 0 \\
0 & 0 & \sigma_3 \\
\end{pmatrix}V^\intercal, \quad
\sigma_1 \geq \sigma_2 \geq \sigma_3 > 0 \tag{4}
$$

### 4. Let the column vectors of matrix $V$ be $v_1, v_2, v_3$, and calculate the planar parameters $(n, h)$ as follows

$Norm$ denotes normalization to the unit vector.

$$
n=Norm\left[\sqrt{\sigma_1^2-\sigma_2^2}v_1\pm \sqrt{\sigma_2^2-\sigma_3^2}v_3\right], \quad h=\frac{\sigma_2}{\sigma_1 - \sigma_3} \tag{5}
$$

### 5. Calculate motion parameters $(t, R)$

$$
t=Norm\left[-\sigma_3\sqrt{\sigma_1^2-\sigma_2^2}v_1\pm \sigma_1\sqrt{\sigma_2^2-\sigma_3^2}v_3\right], \quad R=\frac{1}{\sigma_2} \left(I + \frac{\sigma_2^3nt^\intercal}{h} \right)\tilde{H}^\intercal \tag{6}
$$

### 6. The one that changes the sign of $n$ and $t$ at the same time is also a solution

<br></br>

You can try homography decomposition by running follow command.

```bash
python3 decompose_homography.py
```

The meaning of each formula can be understood from [this book](https://www.morikita.co.jp/books/mid/081791).

<br></br>

## Solution Selection
The method of reducing the number of solutions from four to one can be found in the literature below.

- [Deeper understanding of the homography
decomposition for vision-based control](https://inria.hal.science/inria-00174036/document)

There is also a method where only one solution is calculated from the beginning.

- [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

<br></br>

## Reference
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
