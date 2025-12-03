function [interpr, finterpr] = splinerad( r, f, even )
  % Interpolating odd/even functions along the radius direction
  %
  % Input:
  %
  % r: radial grid. Must be positive and can contain zero.
  % f: value of function on r
  % even: 1 for even function
  %       0 for odd function
  %
  % Output:
  %
  % interpr: interpolation grid r that contains the same number of positive
  % and negative points and does not contain zero, and try to avoid the
  % singular behavior near r=0.
  % fitnerpr: value of the function that is even/odd depends on the "even"
  % parameter.

  stp = 0.0001;
  % rmin avoid r=0 and extrapolation beyond the given grid.
  rmax = ceil(max(r(:)));
  rmin = r(min(find(r>0)));

  interpr = (rmin:stp:rmax)';
  interpr = [-interpr(end:-1:1); interpr];
  pos = find(interpr>0);
  neg = find(interpr<0); neg = neg(end:-1:1);

  spinput = csape(r, f);
  finterpr = zeros(size(interpr));
  finterpr(pos) = fnval(spinput, interpr(pos));
  if( even )
    finterpr(neg) = finterpr(pos);
  else
    finterpr(neg) = -finterpr(pos);
  end
end
