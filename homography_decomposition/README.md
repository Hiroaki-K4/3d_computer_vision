# 3D reconstruction of a plane surface
Shows the procedure for calculating the position of a plane and the position and orientation of a camera from a homography matrix (projective transformation matrix). Since multiple solutions are obtained at this time, we describe the procedure for selecting the correct solution from them.

<br></br>

## Self-calibration by plane
In order to perform planar triangulation, we need the camera matrices $P,P\prime$ of the two cameras and the equation of the plane. We consider a self-calibration that computes this from only the homography matrix $H$ between the two images. The homography matrix is indefinite by a constant factor and has 8 degrees of freedom. The equation of the plane is determined by 3 parameters. Therefore, to self-calibrate, the camera matrices $P,P\prime$ must be expressed in terms of 5 parameters. The unknowns are the two focal lengths $f,f\prime$ and the relative translations $t$ (2 degrees of freedom) and rotations $R$ (3 degrees of freedom) of the camera, for a total of 7 degrees of freedom. To reduce this to 5 DoFs, we assume that the focal lengths $f,f\prime$ are known.

<br></br>

## Formula details

<br></br>

## Reference
- [3D Computer Vision Computation Handbook](https://www.morikita.co.jp/books/mid/081791)
