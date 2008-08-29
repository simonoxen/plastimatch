[a,ainfo] = readmha ('hi_gcs.mha');

wx = 5 * ones(size(a));
wy = zeros(size(a));
wz = zeros(size(a));

w(:,:,:,1) = 5 * ones(size(a));
w(:,:,:,2) = zeros(size(a));
w(:,:,:,3) = zeros(size(a));
writemha ('synth_x_5mm_vf.mha',w,ainfo.Offset,ainfo.ElementSpacing,'float');

w(:,:,:,1) = zeros(size(a));
w(:,:,:,2) = 5 * ones(size(a));
w(:,:,:,3) = zeros(size(a));
writemha ('synth_y_5mm_vf.mha',w,ainfo.Offset,ainfo.ElementSpacing,'float');

w(:,:,:,1) = zeros(size(a));
w(:,:,:,2) = zeros(size(a));
w(:,:,:,3) = 5 * ones(size(a));
writemha ('synth_z_5mm_vf.mha',w,ainfo.Offset,ainfo.ElementSpacing,'float');

[x,y] = meshgrid([1:size(a,1)]+1/2-size(a,1)/2,[1:size(a,2)]+1/2-size(a,2)/2);
d = sin(20*atan2(x,y));
xd = x / size(a,1);
yd = y / size(a,2);
wx = (10 * d .* xd)';
for i=1:size(w,3)
  w(:,:,i,1) = wx;
end
wy = (10 * d .* yd)';
for i=1:size(w,3)
  w(:,:,i,2) = wy;
end
w(:,:,:,3) = zeros(size(a));
writemha ('synth_radial_vf.mha',w,ainfo.Offset,ainfo.ElementSpacing,'float');
