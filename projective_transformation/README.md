# Projective transformation
Suppose a plane is captured by two cameras and a point $(x, y)$ in one image corresponds to a point $(x\prime, y\prime)$ in the other image. At this time, it is known that the following relationship holds.

$$
x\prime=f_0\frac{H_{11}x+H_{12}y+H_{13}f_0}{H_{31}x+H_{32}y+H_{33}f_0},y\prime=\frac{H_{21}x+H_{22}y+H_{23}f_0}{H_{31}x+H_{32}y+H_{33}f_0} \tag{1}
$$

As with elliptic fitting and the computation of the basis matrix, $f_0$ is a constant that adjusts the scale.
Eq(1) can be rewritten as where $\simeq$ denotes that the left-hand side is a non-zero constant multiple of the right-hand side.

$$
\begin{pmatrix}
x\prime/f_0 \\
y\prime/f_0 \\
1 \\
\end{pmatrix}\simeq
\begin{pmatrix}
H_{11} & H_{12} & H_{13} \\
H_{21} & H_{22} & H_{23} \\
H_{31} & H_{32} & H_{33} \\
\end{pmatrix}
\begin{pmatrix}
x/f_0 \\
y/f_0 \\
1 \\
\end{pmatrix} \tag{2}
$$

Such a mapping from $(x,y)$ to $(x\prime,y\prime)$ is called a projective transformation when $H$ is a regular matrix. Eq(2) can be rewritten as follows, since the left and right sides are parallel vectors.

$$
\begin{pmatrix}
x\prime/f_0 \\
y\prime/f_0 \\
1 \\
\end{pmatrix}\times
\begin{pmatrix}
H_{11} & H_{12} & H_{13} \\
H_{21} & H_{22} & H_{23} \\
H_{31} & H_{32} & H_{33} \\
\end{pmatrix}
\begin{pmatrix}
x/f_0 \\
y/f_0 \\
1 \\
\end{pmatrix}
=\begin{pmatrix}
0 \\
0 \\
0 \\
\end{pmatrix} \tag{3}
$$

$\times$ represents the outer product.

The regular matrix $H$ is determined from the relative positions of the two cameras and their parameters as well as the position and orientation of the planar scene and is called the projective transformation matrix. It represents the same projective transformation no matter how many times the whole matrix is multiplied, so it is multiplied by a constant and normalized as follows.

$$
\|H\|\left(\equiv \sqrt{\sum_{i,j=1}^3H_{ij}^2}\right)=1 \tag{4}
$$

The 9-dimensional vector is defined as follows.

$$
\theta=\begin{pmatrix}
H_{11} \\
H_{12} \\
H_{13} \\
H_{21} \\
H_{22} \\
H_{23} \\
H_{31} \\
H_{32} \\
H_{33} \\
\end{pmatrix},
\xi^{(1)}=\begin{pmatrix}
0 \\
0 \\
0 \\
-f_0x \\
-f_0y \\
-f_0^2 \\
xy\prime \\
yy\prime \\
f_0y\prime \\
\end{pmatrix},
\xi^{(2)}=\begin{pmatrix}
f_0x \\
f_0y \\
f_0^2 \\
0 \\
0 \\
0 \\
-xx\prime \\
-yx\prime \\
-f_0x\prime \\
\end{pmatrix},
\xi^{(3)}=\begin{pmatrix}
-xy\prime \\
-yy\prime \\
-f_0y\prime \\
xx\prime \\
yx\prime \\
f_0x\prime \\
0 \\
0 \\
0 \\
\end{pmatrix} \tag{5}
$$

The three components of equation (3) can be written as

$$
(\xi^{(1)},\theta)=0, \qquad (\xi^{(2)},\theta)=0, \qquad (\xi^{(3)},\theta)=0 \tag{6}
$$

<br></br>

## Error and covariance matrix
Corresponding point with error $(x_\alpha,y_\alpha)(x\prime_\alpha,y\prime_\alpha)(\alpha=1,...,N)$ from which a projective transformation matrix $H$ satisfying Eq(1) is computed is mathematically to compute a unit vector a such that

$$
(\xi^{(1)},\theta)\approx 0, \qquad (\xi^{(2)},\theta)\approx 0, \qquad (\xi^{(3)},\theta)\approx 0, \qquad \alpha=1,...,N \tag{7}
$$

$\xi_\alpha^{(k)}$ is the value of $\xi^{(k)}$ for $x=x_\alpha, y=y_\alpha, x\prime=x\prime_\alpha, y\prime=y\prime_\alpha$. The data $x_\alpha, y_\alpha, x\prime_\alpha, y\prime_\alpha$ is written as its true value $\bar{x_\alpha}, \bar{y_\alpha}, \bar{x_\alpha}\prime, \bar{y_\alpha}\prime$ plus the error $\triangle x_\alpha, \triangle y_\alpha, \triangle x_\alpha\prime, \triangle y_\alpha\prime$ as follows.

$$
x_\alpha=\bar{x_\alpha}+\triangle x_\alpha, \quad y_\alpha=\bar{y_\alpha}+\triangle y_\alpha, \quad x_\alpha\prime=\bar{x_\alpha}\prime+\triangle x_\alpha\prime, \quad y_\alpha\prime=\bar{y_\alpha}\prime+\triangle y_\alpha\prime \tag{8}
$$

Substituting these into $\xi_\alpha^{(k)}$, we obtain

$$
\xi_\alpha^{(k)}=\bar{\xi_\alpha^{(k)}}+\triangle_1 \xi_\alpha^{(k)} + \triangle_2 \xi_\alpha^{(k)} \tag{9}
$$

$\bar{\xi_\alpha^{(k)}}$ is the value of $\xi_\alpha^{(k)}$ with respect to $x_\alpha=\bar{x_\alpha}, y_\alpha=\bar{y_\alpha}, x_\alpha\prime=\bar{x_\alpha\prime}, y_\alpha\prime=\bar{y_\alpha\prime}$, and $\triangle_1 \xi_\alpha^{(k)}$ and $\triangle_2 \xi_\alpha^{(k)}$ are the first-order and second-order error terms, respectively. the first-order error term can be written as

$$
\triangle_1 \xi_\alpha^{(k)}=T_\alpha^{(k)}
\begin{pmatrix}
\triangle x_\alpha \\
\triangle y_\alpha \\
\triangle x_\alpha\prime \\
\triangle y_\alpha\prime \\
\end{pmatrix} \tag{10}
$$

The matrix $T_\alpha^{(k)}$ is a sequence of $\xi_\alpha^{(k)}$ differentiated by $x_\alpha$, $y_\alpha$, $x_\alpha\prime$ and $y_\alpha\prime$ respectively, and arranged as a column, as follows.

$$
T_\alpha^{(1)}=\begin{pmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
-f_0 & 0 & 0 & 0 \\
0 & -f_0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\bar{y_\alpha}\prime & 0 & 0 & \bar{x_\alpha} \\
0 & \bar{y_\alpha}\prime & 0 & \bar{y_\alpha} \\
0 & 0 & 0 & f_0 \\
\end{pmatrix},
T_\alpha^{(2)}=\begin{pmatrix}
f_0 & 0 & 0 & 0 \\
0 & f_0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
-\bar{x_\alpha\prime} & 0 & -\bar{x_\alpha} & 0 \\
0 & -\bar{x_\alpha\prime} & -\bar{y_\alpha} & 0 \\
0 & 0 & -f_0 & 0 \\
\end{pmatrix},
T_\alpha^{(3)}=\begin{pmatrix}
-\bar{y_\alpha\prime} & 0 & 0 & -\bar{x_\alpha} \\
0 & -\bar{y_\alpha\prime} & 0 & -\bar{y_\alpha} \\
0 & 0 & 0 & -f_0 \\
\bar{x_\alpha\prime} & 0 & \bar{x_\alpha} & 0 \\
0 & \bar{x_\alpha\prime} & \bar{y_\alpha} & 0 \\
0 & 0 & f_0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{pmatrix} \tag{11}
$$

Considering the error $\triangle x_\alpha, \triangle y_\alpha, \triangle x_\alpha\prime, \triangle y_\alpha\prime$ as a random variable, we define the covariance matrix of $\xi_\alpha^{(k)}$ and $\xi_\alpha^{(l)}$ as follows.

$$
V^{(kl)}[\xi_\alpha]=E[\triangle_1 \xi_\alpha^{(k)} \triangle_1 \xi_\alpha^{(l)\intercal}] \tag{12}
$$

$E[・]$ denotes the expected value for that distribution. If $\triangle x_\alpha, \triangle y_\alpha, \triangle x_\alpha\prime, \triangle y_\alpha\prime$ follow a normal distribution with expectation $0$ and standard deviation $\sigma$, independent of each other, then

$$
\begin{align*}
E[\triangle x_\alpha]&=E[\triangle y_\alpha]=E[\triangle x_\alpha\prime]=E[\triangle y_\alpha\prime]=0, \\
E[\triangle x_\alpha^2]&=E[\triangle y_\alpha^2]=E[\triangle x_\alpha\prime^2]=E[\triangle y_\alpha\prime^2]=\sigma^2, \\
E[\triangle x_\alpha\triangle y_\alpha]&=E[\triangle x_\alpha\prime\triangle y_\alpha\prime]=E[\triangle x_\alpha\triangle y_\alpha\prime]=E[\triangle x_\alpha\prime\triangle y_\alpha]=0
\end{align*} \tag{13}
$$

The covariance matrix of Eq(12), using Eq(10), can be written as

$$
V^{(kl)}[\xi_\alpha]=\sigma^2V_0^{(kl)}[\xi_\alpha], \quad V_0^{(kl)}[\xi_\alpha]=T_\alpha^{(k)}T_\alpha^{(l)\intercal} \tag{14}
$$

As with elliptic fitting, there is no need to consider $\triangle_2\xi_\alpha$. Also, $\bar{x_\alpha},\bar{y_\alpha},\bar{x_\alpha\prime},\bar{y_\alpha\prime}$ in Eq(11) is replaced by $x_\alpha,y_\alpha,x_\alpha\prime,y_\alpha\prime$ in the actual calculation.

<br></br>

## Algebraic method
To compute the projective transformation matrix $H$ from the corresponding points with errors is to compute the unit vector a satisfying Eq.(7), taking into account the nature of the errors represented by the covariance matrix $V^{(kl)}[\xi_\alpha]$.

### Weighted repetition method
#### **1. Let $\theta=0$ and $W_\alpha^{(kl)}=\delta_{kl}$. $\delta_{kl}$ is the Kronecker delta ($1$ when $k=l$, $0$ otherwise).**

#### **2. Calculate 9*9 matrix M**

$$
M = \frac{1}{N}\sum_{\alpha=1}^N\sum_{k,l=1}^3 W_\alpha^{(kl)}\xi_\alpha^{(k)}\xi_\alpha^{(l)\intercal} \tag{15}
$$

#### **3. Solve the following eigenvalue problem to compute the unit eigenvector $\theta$ for the smallest eigenvalue $\lambda$**

$$
M\theta=\lambda\theta \tag{16}
$$

#### **4. If $\theta\approx\theta_0$, except for the sign, return $\theta$ and exit. Otherwise, update as follows and return to step 2.**

$$
W_\alpha^{(kl)} \leftarrow \left( (\theta, V_0^{(kl)}[\xi_\alpha]\theta) \right)_2^-, \quad \theta_0 \leftarrow \theta \tag{17}
$$

$\left( (\theta, V_0^{(kl)}[\xi_\alpha]\theta) \right)_2^-$ in Eq.(17) is the $(k,l)$ element of the rank 2 general inverse of the matrix with $(\theta, V_0^{(kl)}[\xi_\alpha]\theta)$ as the $(k,l)$ element. In other words, it is an abbreviation for the following matrix.

$$
\begin{pmatrix}
(\theta, V_0^{(11)}[\xi_\alpha]\theta) & (\theta, V_0^{(12)}[\xi_\alpha]\theta) & (\theta, V_0^{(13)}[\xi_\alpha]\theta) \\
(\theta, V_0^{(21)}[\xi_\alpha]\theta) & (\theta, V_0^{(22)}[\xi_\alpha]\theta) & (\theta, V_0^{(23)}[\xi_\alpha]\theta) \\
(\theta, V_0^{(31)}[\xi_\alpha]\theta) & (\theta, V_0^{(32)}[\xi_\alpha]\theta) & (\theta, V_0^{(33)}[\xi_\alpha]\theta) \\
\end{pmatrix}_2^- \tag{18}
$$

The general inverse matrix $A_r^-$ of rank $r$ of a symmetric matrix $A$ is the orthonormal system of eigenvalues of $A$ with $\lambda_1 \geq \lambda2 \geq ...$ and corresponding unit eigenvectors $u1,u2,...$ Then $A_r^-=u_1u_1^\intercal/\lambda_1+...+u_2u_2^\intercal/\lambda_2$. Similar to elliptic fitting, the computation of the unit eigenvector for the smallest eigenvalue of a symmetric matrix $M$ is also the computation of the unit vector $\theta$ that minimizes the quadratic form $(\theta, M\theta)$.

$$
\begin{align*}
(\theta,M\theta)&=\left( \theta, \left( \frac{1}{N}\sum_{k,l=1}^3 W_\alpha^{(kl)}\xi_\alpha^{(k)}\xi_\alpha^{(l)\intercal} \right) \theta \right) \\
&=\frac{1}{N}\sum_{k,l=1}^3 W_\alpha^{(kl)}(\theta, \xi_\alpha^{(k)}\xi_\alpha^{(l)\intercal}\theta) \\
&=\frac{1}{N}\sum_{k,l=1}^3 W_\alpha^{(kl)}(\theta, \xi_\alpha^{(k)})(\theta, \xi_\alpha^{(l)})
\end{align*} \tag{19}
$$

This is called the weighted least squares method. According to statistics, it is known that the weight $W_\alpha^{(kl)}$ is best taken to be proportional to the elements of the inverse of the covariance matrix for that term (thus, the term with the smallest error is larger and the term with the largest error is smaller).

Since $(\xi_\alpha^{(k)}, \theta)=(\bar{\xi_\alpha^{(k)}}, \theta)+(\triangle_1 \xi_\alpha^{(k)}, \theta) + (\triangle_2 \xi_\alpha^{(k)}, \theta)$ and $(\bar{\xi_\alpha^{(k)}}, \theta)=0$, the covariance matrix follows from Eq(12) and Eq(14).

$$
E[(\triangle_1 \xi_\alpha^{(k)}, \theta)(\triangle_1 \xi_\alpha^{(k)}, \theta)]=(\theta, E[\triangle_1 \xi_\alpha^{(k)} \triangle_1 \xi_\alpha^{(l)^\intercal}]\theta)=\sigma^2(\theta,V_0^{(kl)}[\xi_\alpha]\theta) \tag{20}
$$

Therefore, we can take the $(k,l)$ elements of the inverse matrix $(\theta,V_0^{(kl)}[\xi_\alpha]\theta)$ as weights $W_\alpha^{(kl)}$. However, when there is no error in the data, $(\theta,V_0^{(kl)}[\xi_\alpha]\theta)$ has determinant zero and there is no inverse matrix. This is because the three equations in Eq(6) are linearly dependent.
In fact, from Eq(5), $x\prime\xi^{(1)}-y\prime\xi^{(2)}=\xi^{(3)}$. Therefore, the three equations in Eq(6) are essentially useless, e.g., if equations 1 and 2 are available, then equation 3 is automatically satisfied. Therefore, only two equations should be used, but which two are specified is not unique. To get away without specifying, we can use a rank 2 general inverse matrix instead of an inverse matrix.

The reason why step 4 says except for the sign is because the general eigenvector has an indefinite sign. For this reason, if $(\theta, \theta_0)<0$, the comparison is done by changing the sign from $\theta$ to $-\theta$.

You can try weighted repetition by running below command.

```bash
python3 calculate_projective_trans_by_weighted_repetition.py
```

<img src='images/img_0.png' width='350'><img src='images/img_1.png' width='350'>

<br></br>

## Reference
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
