# Bundle adjustment

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

You can get camera intrinsic parameter $K$, rotation $R$ and translation $t$ by running below command. 3D positions obtained by decomposing camera parameters from the [oxford dinosaur dataset](https://www.robots.ox.ac.uk/~vgg/data/mview/) are drawn as the below image.

```bash
python3 disassemble_camera_matrix.py
```

<img src='ref_images/disassemble.png' width='600'>

The oxford dinosaur dataset is as follows.

<img src='ref_images/dinosaur.jpg' width='600'>

<br></br>

## References
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
- [Multi-view Data](https://www.robots.ox.ac.uk/~vgg/data/mview/)
