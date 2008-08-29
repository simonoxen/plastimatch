function ct = get_cms_ct_dir(dirname)
% function ct = get_cms_ct_dir(dirname)

d = dir([dirname, '/*.CT']);
for i = 1:length(d)
  z(i) = str2num(d(i).name(3:end-3));
end
[y,idx] = sort(z);
for i = 1:length(d)
  j = idx(i);
  disp(['reading ', d(j).name]);
  ct1 = get_cms_ct([dirname, '/', d(j).name]);
  if (i==1)
    ct = ct1;
    ct.img = zeros(size(ct1.img,1),size(ct1.img,2),length(d));
    ct.img(:,:,1) = ct1.img;
  else
    ct.zpos = [ct.zpos; ct1.zpos];
    ct.img(:,:,i) = ct1.img;
  end
end

if (min(diff(ct.zpos)) ~= max(diff(ct.zpos)))
  disp('Warning: uneven slice thicknesses');
end

