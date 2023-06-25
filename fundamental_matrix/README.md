# Fundamental matrix
Suppose two cameras capture the same scene, and a certain point in the scene is captured at point (x, y) in the image captured by the first camera and at point (x', y') in the image captured by the second camera. The following relation holds between them.

$$
\begin{align*}
\begin{pmatrix}
\begin{pmatrix}
x/f_0 \\
y/f_0 \\
1 \\
\end{pmatrix},
\begin{pmatrix}
x\prime/f_0 \\
y\prime/f_0 \\
1 \\
\end{pmatrix}
\end{pmatrix}=0 \tag{1}
\end{align*}
$$

$f_0$ is a constant that adjusts the scale as in elliptic fitting. For numerical calculations on a finite-length computer, it is convenient for $x/f0$ and $y/f0$ to be about 1.  
$F=(F_{ij})$ is a 3x3 matrix determined from the relative positions of the two cameras and their parameters (focal length, etc.) and is called the fundamental matrix. Eq(1) is called the epipolar constraint or the epipolar equation. Eq(1) represents the same equation no matter how many times $F$ is multiplied, so multiply it by a constant and normalized as follows.

$$
\begin{align*}
||F||=\sqrt{\sum_{i,j=1}^3F_{ij}^2}=1 \tag{2}
\end{align*}
$$

If we define a 9-dimensional vector as follows,

$$
\begin{align*}
\xi=\begin{pmatrix}
xx\prime \\
xy\prime \\
f_0x \\
yx\prime \\
yy\prime \\
f_0y \\
f_0x\prime \\
f_0y\prime \\
f_0^2 \\
\end{pmatrix},
\theta=\begin{pmatrix}
F_{11} \\
F_{12} \\
F_{13} \\
F_{21} \\
F_{22} \\
F_{23} \\
F_{31} \\
F_{32} \\
F_{33} \\
\end{pmatrix} \tag{3}
\end{align*}
$$

It can be seen that the left-hand side of Eq(1) is $(\xi,\theta)/f_0^2$. Hence, the epipolar equation in Eq(1) can also be written as

$$
\begin{align*}
(\xi,\theta)=0 \tag{4}
\end{align*}
$$

Thus, in form, Eq(4) is equivalent to elliptic fitting.

<br></br>

## Covariance matrix and algebraic methods
Calculating the fundamental matrix $F$ satisfying Eq(1) from the corresponding points $(x_\alpha,y_\alpha), (x_\alpha\prime,y_\alpha\prime), (\alpha=1,...,N)$ with errors is mathematically to compute a unit vector $\theta$ such that

$$
\begin{align*}
(\xi_\alpha,\theta)\approx0, \alpha=1,...,N \tag{5}
\end{align*}
$$

$\xi_\alpha$ is the value of $\xi$ for $x=x_\alpha,y=y_\alpha,x\prime=x_\alpha\prime,y\prime=y_\alpha\prime$ in Eq(3).
Data $x_\alpha,y_\alpha,x_\alpha\prime,y_\alpha\prime$ can be written as its true value $\bar{x_\alpha},\bar{y_\alpha},\bar{x_\alpha\prime},\bar{y_\alpha\prime}$ plus an error $\triangle{x_\alpha},\triangle{y_\alpha},\triangle{x_\alpha\prime},\triangle{y_\alpha\prime}$ as follows.

$$
\begin{align*}
x_\alpha=\bar{x_\alpha}+\triangle{x_\alpha},y_\alpha=\bar{y_\alpha}+\triangle{y_\alpha},x_\alpha\prime=\bar{x_\alpha\prime}+\triangle{x_\alpha\prime},y_\alpha\prime=\bar{y_\alpha\prime}+\triangle{y_\alpha\prime} \tag{6}
\end{align*}
$$

Substituting this into $\xi_\alpha$ obtained from Eq(3), we obtain

$$
\begin{align*}
\xi_\alpha=\bar{\xi_\alpha}+\triangle_1\xi_\alpha+\triangle_2\xi_\alpha \tag{7}
\end{align*}
$$

where $\bar{\xi_\alpha}$ is the true value and $\triangle_1\xi_\alpha$ and $\triangle_2\xi_\alpha$ are the first and second order error terms, respectively. Expanding it , we obtain the following.

$$
\begin{align*}
\triangle_1\xi_\alpha=\begin{pmatrix}
\bar{x_\alpha}\prime\triangle x_\alpha+\bar{x_\alpha}\triangle x_\alpha\prime \\
\bar{y_\alpha}\prime\triangle x_\alpha+\bar{x_\alpha}\triangle y_\alpha\prime \\
f_0\triangle x_\alpha \\
\bar{x_\alpha}\prime\triangle y_\alpha+\bar{y_\alpha}\triangle x_\alpha\prime \\
\bar{y_\alpha}\prime\triangle y_\alpha+\bar{y_\alpha}\triangle y_\alpha\prime \\
f_0\triangle y_\alpha \\
f_0\triangle x_\alpha\prime \\
f_0\triangle y_\alpha\prime \\
0 \\
\end{pmatrix},
\triangle_2\xi_\alpha=\begin{pmatrix}
\triangle x_\alpha\triangle x_\alpha\prime \\
\triangle x_\alpha\triangle y_\alpha\prime \\
0 \\
\triangle y_\alpha\triangle x_\alpha\prime \\
\triangle y_\alpha\triangle y_\alpha\prime \\
0 \\
0 \\
0 \\
0 \\
\end{pmatrix} \tag{8}
\end{align*}
$$

Considering the errors $\triangle x_\alpha$ and $\triangle y_\alpha$ as random variables, the covariance matrix of $\xi_\alpha$ is defined as follows.

$$
\begin{align*}
V[\xi_\alpha]=E[\triangle_1\xi_\alpha\triangle_1\xi_\alpha^T] \tag{9}
\end{align*}
$$

$E$ represents the expected value of that distribution. if $\triangle x_\alpha$ and $\triangle y_\alpha$ follow a normal distribution with expectation 0 and standard deviation $\sigma$, independent of each other, then the following equation holds

$$
\begin{align*}
E[\triangle x_\alpha]&=E[\triangle y_\alpha]=E[\triangle x_\alpha\prime]=E[\triangle y_\alpha\prime]=0, \\ \tag{10}
E[\triangle x_\alpha^2]&=E[\triangle y_\alpha^2]=E[\triangle x_\alpha\prime^2]=E[\triangle y_\alpha\prime^2]=\sigma^2, \\
E[\triangle x_\alpha\triangle y_\alpha]&=E[\triangle x_\alpha\prime\triangle y_\alpha\prime]=E[\triangle x_\alpha\triangle y_\alpha\prime]=E[\triangle x_\alpha\prime\triangle y_\alpha]=0
\end{align*}
$$

Using Eq(8), Eq(9) can be written as follows

$$
\begin{align*}
V[\xi_\alpha]=\sigma^2V_0[\xi_\alpha] \tag{11}
\end{align*}
$$

However, since all the elements are multiplied by $\sigma^2$, the next matrix from which they are taken is written as $V_0[\xi_\alpha]$ and called the normalized covariance matrix.

$$
\begin{align*}
V_0[\xi_\alpha]=\begin{pmatrix}
\bar{x_\alpha}^2+\bar{x_\alpha\prime}^2 & \bar{x_\alpha\prime}\bar{y_\alpha\prime} & f_0\bar{x_\alpha\prime} & \bar{x_\alpha}\bar{y_\alpha} & 0 & 0 & f_0\bar{x_\alpha} & 0 & 0 \\
\bar{x_\alpha\prime}\bar{y_\alpha\prime} & \bar{x_\alpha}^2+\bar{y_\alpha\prime}^2 & f_0\bar{y_\alpha\prime} & 0 & \bar{x_\alpha}\bar{y_\alpha} & 0 & 0 & f_0\bar{x_\alpha}  & 0 \\
f_0\bar{x_\alpha\prime} & f_0\bar{y_\alpha\prime} & f_0^2 & 0 & 0 & 0 & 0 & 0 & 0 \\
\bar{x_\alpha}\bar{y_\alpha} & 0 & 0 & \bar{y_\alpha}^2+\bar{x_\alpha\prime}^2 & \bar{x_\alpha\prime}\bar{y_\alpha\prime} & f_0\bar{x_\alpha\prime} & f_0\bar{y_\alpha\prime} & 0 & 0 \\
0 & \bar{x_\alpha}\bar{y_\alpha} & 0 & \bar{x_\alpha\prime}\bar{y_\alpha\prime} & \bar{y_\alpha}^2+\bar{y_\alpha\prime}^2 & f_0\bar{y_\alpha\prime} & 0 & f_0\bar{y_\alpha} & 0 \\
0 & 0 & 0 & f_0\bar{x_\alpha\prime} & f_0\bar{y_\alpha\prime} & f_0^2 & 0 & 0 & 0 \\
f_0\bar{x_\alpha} & 0 & 0 & f_0\bar{y_\alpha} & 0 & 0 & f_0^2 & 0 & 0 \\
0 & f_0\bar{x_\alpha} & 0 & 0 & f_0\bar{y_\alpha} & 0 & 0 & f_0^2 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix} \tag{12}
\end{align*}
$$

As with elliptic fitting, there is no need to consider $\triangle_2\xi_\alpha$ in the covariance matrix in Eq(9). Also, $\bar{x_\alpha},\bar{y_\alpha},\bar{x_\alpha\prime},\bar{y_\alpha\prime}$ in Eq(12) is replaced by $x_\alpha,y_\alpha,x_\alpha\prime,y_\alpha\prime$ in the actual calculation.  
To compute the fundamental matrix $F$ from the corresponding points with errors is to compute $θ$ satisfying Eq(5), taking into account the nature of the errors. This is the same as for elliptic fitting.

<br></br>

## **Solution1. Least squares method**

### **1. Calculate 9x9matrix $M$**
$$
\begin{align*}
M=\frac{1}{N}\sum_{\alpha=1}^N\xi_\alpha\xi_\alpha^\intercal \tag{13}
\end{align*}
$$

### **2. Solve the eigenvalue problem and return the unit eigenvector $\theta$ for the smallest eigenvalue $\lambda$**
$$
\begin{align*}
M\theta=\lambda\theta \tag{14}
\end{align*}
$$

Minimize the following sum of squares under condition $||\theta||=1$.

<br></br>

## **Solution2. Taubin method**
### **1. Calculate 9x9matrix $M$ and $N$**
$$
\begin{align*}
M=\frac{1}{N}\sum_{\alpha=1}^N\xi_\alpha\xi_\alpha^\intercal,
N=\frac{1}{N}\sum_{\alpha=1}^NV_0[\xi_\alpha] \tag{15}
\end{align*}
$$

### **2. Solve the eigenvalue problem and return the unit eigenvector $\theta$ for the smallest eigenvalue $\lambda$**
$$
\begin{align*}
M\theta=\lambda N\theta \tag{16}
\end{align*}
$$

<br></br>

## Code
You can try the calculation of fundamental matrix by running below command.

```bash
python3 calculate_f_matrix.py
```

<br></br>

## Reference
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
