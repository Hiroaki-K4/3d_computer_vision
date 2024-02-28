# Bundle adjustment
Bundle adjustment is a technique that calculates the 3D shapes of the scene, positions of each camera and the intrinsic parameters simultaneously to satisfy the perspective projection model of camera by using images taken with multiple cameras of a 3D scene. We fix one world coordinates and let the position of $k$-th camera $(k=1,...,M)$ be $t_k$. And, we suppose that the coordinates of each camera is rotated by $R_k$ with respect to the world coordinates and let the focal length be $f_k$ and the optical axis point be $(u_{0k},v_{0k})$.  
We define the projection positionof $\alpha$-th point $(X_\alpha, Y_\alpha, Z_\alpha)$ on $k$-th image as $(x_{\alpha k}, y_{\alpha k})$. The relationship of the perspective projection is as follows.

$$
\begin{pmatrix}
x_{\alpha k}/f_0 \\
y_{\alpha k}/f_0 \\
1 \\
\end{pmatrix}\simeq
P_k
\begin{pmatrix}
X_\alpha \\
Y_\alpha \\
Z_\alpha \\
1 \\
\end{pmatrix} \tag{1}
$$

The camera matrix $P_k$ of $k$-th camera has following form.

$$
P_k=
\begin{pmatrix}
f_k & 0 & u_{0k} \\
0 & f_k & v_{0k} \\
0 & 0 & f_0 \\
\end{pmatrix}
\begin{pmatrix}
R_k^\intercal & -R_k^\intercal t_k \\
\end{pmatrix} \tag{2}
$$



<br></br>

# Experiments
## Prepare dataset
We need to prepare the dataset for bundle adjustment by following steps. We use the [Oxford dinosaur dataset](https://www.robots.ox.ac.uk/~vgg/data/mview/).

### 1. Convert ppm images to jpg images.

```bash
python3 convert_ppm_to_jpg.py
```

### 2. Read camera matrix from matlab file and save it to json file.

```bash
python3 read_matlab_file.py
```

### 3. Disassemble camera to matrix to get camera intrinsic parameter, rotation matrix and translation matrix.

```bash
python3 disassemble_camera_matrix.py
```

### 4. Calculate initial 3D position of points by triangulation.

```bash
python3 calculate_3d_points_by_triangulation.py
```

<img src='ref_images/points_3d.png' width='600'>

<br></br>

## Appendix: Camera matrix decomposition
We can calculate camera intrinsic parameter $K$, rotation $R$ and translation $t$ when $3\times 4$ camera matrix $P$ is given. The method is as follows.

### 1. Define $P_k$ as $P_k=\begin{pmatrix}Q & q\end{pmatrix}$. That is, define first $3\times 3$ part of $P_k$ as $Q$ and 4th column as $q$.

### 2. If $detQ < 0$, change sign of $Q$ and $q$.

### 3. Define translation $t_k$ as follows.

$$
t_k=-Q^{-1}q
$$

### 4. Perform the Cholesky decomposition of $(QQ)^{-1}$ as follows. $C$ is the upper triangular matrix.

$$
(QQ)^{-1}=C^\intercal C
$$

### 5. Define $K_k$ as follows.

$$
K_k=C^{-1}
$$

### 6. Define rotation $R_k$ as follows.

$$
R_k = Q^\intercal C^\intercal
$$

### Explanation
We want to calculate the upper triangular matrix $K_k$, the rotation matrix $R_k$ and the translation vector $t_k$ satisfy following equation.

$$
Q=K_kR_K^\intercal, \quad q=K_kR_K^\intercal t_k
$$

First, $t_k$ is determined as step3 by $q=-Qt_k$. From $R_k^\intercal R_k=I$, we get following relationship.

$$
QQ^\intercal=K_kK_k^\intercal
$$

This inverse matrix is as follows.

$$
(QQ^\intercal)^{-1}=(K_k^-)^\intercal K_k^{-1}
$$

Perform the Cholesky decomposition of $(QQ^\intercal)^{-1}$ and represents as step4 by the upper triangular matrix $C$, we can get $K_k$. And, we can represent $Q$ as follows.

$$
Q=C^{-1}R^\intercal
$$

Transposing both sides, we get $Q^\intercal=R_k(C^\intercal)^{-1}$ and find $R_k$ as step6. $R_k$ calculated like this is the rotation matrix.

$$
R_kR_k^\intercal=Q^\intercal C^\intercal CQ=Q^\intercal(QQ^\intercal)^{-1}Q=Q^\intercal(Q^\intercal)^{-1}Q^{-1}Q=I
$$

$R_k$ satisfies $detR_k > 0$. Because $detR_k$ satisfies $detR_K=detQ^\intercal detC^\intercal=detQdetC$. By adjusting the sign, $detQ>0$ holds true. And $detC>0$ holds true because the signs of the diagonal elements are chosen to be positive in the Cholesky decomposition.

You can get camera intrinsic parameter $K$, rotation $R$ and translation $t$ by running below command. 3D positions obtained by decomposing camera parameters from the [Oxford dinosaur dataset](https://www.robots.ox.ac.uk/~vgg/data/mview/) are drawn as the below image.

```bash
python3 disassemble_camera_matrix.py
```

<img src='ref_images/disassemble.png' width='600'>

The Oxford dinosaur dataset is as follows.

<img src='ref_images/dinosaur.jpg' width='600'>

<br></br>

# References
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
- [Multi-view Data](https://www.robots.ox.ac.uk/~vgg/data/mview/)
