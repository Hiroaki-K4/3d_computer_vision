# Triangulation
If two cameras are used to capture the same scene, the 3D positions of corresponding points can be recovered if the position, orientation, and internal parameters of each camera are known. This is because the line of sight at that point, starting from the lens center of each camera, can be determined, and the triangle connecting the lens centers of the two cameras and that point in the scene can be calculated. This is called triangulation.

<br></br>

# Perspective projection
The XYZ coordinates are fixed in the scene and are called the world coordinate system. On the other hand, apart from that, consider an XcYcZc coordinate system fixed to the camera with the center of the camera lens as Oc and the optical axis as Zc axis. This is called the camera coordinate system. Also, consider a plane with an xy coordinate system fixed to the camera. This is identified with the image and is called the image plane.
Then, a point (X, Y, Z) in the scene is assumed to be captured at the intersection (x, y) of the image plane and a line passing through the point and the lens center Oc. Such a model is called a perspective projection. The lens center Oc is called the viewpoint, the line passing through it and the point (x, y) is called the line of sight, and the intersection point (u0, v0) between the image plane and the optical axis is called the principal point.
Let t be a vector representing the position of viewpoint Oc in the world coordinate system and R be a rotation matrix of the XcYcZc camera coordinate system relative to the XYZ world coordinate system, then the camera position and orientation can be specified by {t,R}. These are called the motion parameters of the camera, and t and R are called the translation and rotation of this camera, respectively.

<br></br>

# Camera Matrix and Triangulation
The point (X,Y,Z) in the scene that is projected onto the image (x,y) is represented by the following fractional expression from the model of perspective projection.

$$
\begin{align*}
x=f_0\frac{P_{11}X+P_{12}Y+P_{13}Z+P_{14}}{P_{31}X+P_{32}Y+P_{33}Z+P_{34}}, y=f_0\frac{P_{21}X+P_{22}Y+P_{23}Z+P_{24}}{P_{31}X+P_{32}Y+P_{33}Z+P_{34}} \tag{1}
\end{align*}
$$

$f_0$ is a scale constant and $Pij(i=1,2,3,j=1,.... ,4)$ are coefficients determined from the camera's internal constants and motion parameters. Eq(1) can be rewritten as

$$
\begin{align*}
\begin{pmatrix}
x/f_0 \\
y/f_0 \\
1 \\
\end{pmatrix}\simeq
\begin{pmatrix}
P_{11} & P_{12} & P_{13} & P_{14} \\
P_{21} & P_{22} & P_{23} & P_{24} \\
P_{31} & P_{32} & P_{33} & P_{34} \\
\end{pmatrix}
\begin{pmatrix}
X \\
Y \\
Z \\
1 \\
\end{pmatrix} \tag{2}
\end{align*}
$$

The matrix $P=(P_{ij})$ is called the camera matrix. To perform triangulation, the camera matrix $P$ must be calculated in advance. This is called camera calibration.
Let $P=(P_{ij})$ and $P\prime=(P\prime_{ij})$ for the two camera matrices, respectively. When a point (X,Y,Z) in the scene is observed at a point (x,y),(x',y') on each image, if there is no error in the observation, (X,Y,Z) can be calculated from (x,y),(x',y') as follows.

## Triangulation by Camera Matrix

### **1. Calculate the following 4x3 matrix T and 4-dimensional vector p**

$$
\begin{align*}
T=\begin{pmatrix}
f_0P_{11}-xP_{31} & f_0P_{12}-xP_{32} & f_0P_{13}-xP_{33} \\
f_0P_{21}-yP_{31} & f_0P_{22}-yP_{32} & f_0P_{23}-yP_{33} \\
f_0P\prime_{11}-x\prime P\prime_{31} & f_0P\prime_{12}-x\prime P\prime_{32} & f_0P\prime_{13}-x\prime P\prime_{33} \\
f_0P\prime_{21}-y\prime P\prime_{31} & f_0P\prime_{22}-y\prime P\prime_{32} & f_0P\prime_{23}-y\prime P\prime_{33} \\
\end{pmatrix},
p=\begin{pmatrix}
f_0P_{14}-xP_{34} \\
f_0P_{24}-yP_{34} \\
f_0P\prime_{14}-x\prime P\prime_{34} \\
f_0P\prime_{24}-y\prime P\prime_{34} \\
\end{pmatrix} \tag{3}
\end{align*}
$$

### **2. Solve the following simultaneous linear equations to obtain X, Y, and Z**

$$
T^\intercal T\begin{pmatrix}
X \\
Y \\
Z \\
\end{pmatrix}=-T^\intercal p \tag{4}
$$

<br></br>

## Reference
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
