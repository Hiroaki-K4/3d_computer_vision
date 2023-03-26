# Elliptic fitting
When a circular object in a scene is photographed, it becomes an ellipse on the image plane, and the 3D position of the object can be analyzed from its shape. Therefore, fitting an ellipse to a sequence of points extracted from an image is one of the basic processes for various applications, including camera calibration and visual robot control.

<br></br>
## Elliptic Formula
The usual formula for an ellipse is as follows.  
Let the center of the ellipse be (Xc, Yc), the length along the x-axis be a, the length along the y-axis be b, and the slope of the ellipse be θ.

$$
\frac{((X-X_c)cos\theta+(Y-Y_c)sin\theta)^2}{a^2}+\frac{(-(X-X_c)sin\theta+(Y-Y_c)cos\theta)^2}{b^2}=1
$$

<br></br>

<img src='../images/ellipse.png' width='400'>


<br></br>
Let's draw circle and elliptic!

```bash
python3 draw_elliptic.py
```

<img src='../images/draw_elliptic.png' width='400'>

<br></br>
## References
- [Elliptic approximation by the least-squares method](https://imagingsolution.blog.fc2.com/blog-entry-20.html)
