# Self-calibration of perspective camera
We describe a method for self-calibrating a perspective projection camera.
First, we show that if we give an unknown called **projective depth** to each point, we can apply the same **factorization method** as in the case of an affine camera. This projective depth is determined so that the observation matrix can be factorized. As a calculation method, we implement a **primary method** that iteratively determines the projective depth so that the column rank of the observation matrix becomes 4.

Imaging by a perspective projection camera is described by the following equation. The unknown constant $z_{\alpha k}$ is called the projective depth.

$$
z_{\alpha k}
\begin{pmatrix}
x_{\alpha k}/f_0 \\
y_{\alpha k}/f_0 \\
1 \\
\end{pmatrix}\simeq
P_kX_\alpha \tag{1}
$$

$X_\alpha$ on the right side is a four-dimensional vector in which the three-dimensional coordinates of the $\alpha$ point $X_\alpha, Y_\alpha, Z_\alpha$ and a constant $1$ are arranged, but if we introduce an unknown projective depth $z_{\alpha k}$, there is no need to consider the condition that the fourth component of vector $X_\alpha$ is $1$. This is because multiplying $X_\alpha$ by a constant is the same as multiplying $z_{\alpha k}$ by a constant.  
Therefore, the position of a point in three-dimensional space is expressed by the ratio of the components of $X$, $X_{\alpha (1)}:X_{\alpha (2)}:X_{\alpha (3)}:X_{\alpha (4)}$. This is called homogeneous coordinates. The actual three-dimensional position $(X_\alpha, Y_\alpha, Z_\alpha)$ is calculated as follows.

$$
X_\alpha=\frac{X_{\alpha (1)}}{X_{\alpha (4)}}, \quad Y_\alpha=\frac{X_{\alpha (2)}}{X_{\alpha (4)}}, \quad Z_\alpha=\frac{X_{\alpha (3)}}{X_{\alpha (4)}} \tag{2}
$$

When the fourth component $X_{\alpha (4)}$ of $X_\alpha$ is $0$, it is interpreted that $X_\alpha$ represents a point at infinity in the $(X_\alpha, Y_\alpha, Z_\alpha)$ direction.

Self-calibration of a perspective projection camera means calculating the simultaneous coordinates $X_\alpha(\alpha=1,...,N)$ of all points and the camera matrix $P_k(k=1,...,M)$ of all cameras from the observation point $(x_{\alpha k},y_{\alpha k})(\alpha=1,...,N,k=1,...,M)$.  
However, $P_k,X_\alpha$ satisfying Eq(1) is not unique. This is because $P'_kX'_\alpha=P_kX_\alpha$ holds even if the following transformation is performed using any $4\times 4$ regular matrix $H$.

$$
P'_k=P_kH, \quad X'_\alpha=H^-X_\alpha \tag{3}
$$


