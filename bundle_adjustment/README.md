2D tracked points
-----------------

The format is:

row1 : x1 y1 x2 y2 ... x36 y36 for track 1
row2 : x1 y1 x2 y2 ... x36 y36 for track 2
etc

i.e. a row gives the 2D measured position of a point as it is tracked
through frames 1 to 36.  If there is no match found in a view then x
and y are -1.

Each row corresponds to a different point.  There are 4838 tracks.

Here are some some useful matlab commands for pulling out and plotting
individual tracks:

load viff.xy;

i = find(viff== -1); % find where ends of tracks marked - indicator var
viff(i) = nan;       % changes these to nan

plot(viff(1,1:2:72),viff(1,2:2:72)) % plots track for point 1
plot(viff(1:end,1:2:72)',viff(1:end,2:2:72)') % plots all tracks
axis ij  % y in -ve direction

x = viff(1:end,1:2:72)';  % pull out x coord of all tracks
y = viff(1:end,2:2:72)';  % pull out y coord of all tracks

m = finite(x);  % selects tracks apart from nans

i = sum(m) > 6; % tracks longer than 6 views
plot(x(:,i),y(:,i)) 
axis ij

Thanks to Andrew Fitzgibbon and Andrew Zisserman
