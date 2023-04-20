# Fundamental matrix
Suppose two cameras capture the same scene, and a certain point in the scene is captured at point (x, y) in the image captured by the first camera and at point (x', y') in the image captured by the second camera. The following relation holds between them.

$$
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
\end{pmatrix}=0
$$

$f_0$ is a constant that adjusts the scale as in elliptic fitting. For numerical calculations on a finite-length computer, it is convenient for $x/f0$ and $y/f0$ to be about 1.