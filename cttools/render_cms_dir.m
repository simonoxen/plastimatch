%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% GCS FIXED 9/21/2006 -- not always a dicom dir
if (dicomdir)
  d = dir(dicomdir);
  dicomfn = [dicomdir '/' d(3).name];

  %% Get (x,y) coordinates for the mask
  info = dicominfo(dicomfn);
  I = dicomread(info);
  ps = info.PixelSpacing;
  ipp = info.ImagePositionPatient;
  cr = [info.Columns,info.Rows];

% x = ipp(1):ps(1):ipp(1)+ps(1)*cr(1);
% y = ipp(2):ps(2):ipp(2)+ps(2)*cr(2);

else
  d = dir([cmsdir,'/*.CT']);
  ct1 = get_cms_ct([cmsdir, '/', d(1).name]);
  
  ipp = [ct1.xoff, ct1.yoff];
  cr = [ct1.xres, ct1.yres];
  ps = [ct1.xpixsize, ct1.ypixsize];
end

x = ipp(1)+ps(1)/2:ps(1):ipp(1)+ps(1)*cr(1);
y = ipp(2)+ps(2)/2:ps(2):ipp(2)+ps(2)*cr(2);

contournames_fn = [maskdir '/contournames'];

d = dir(maskdir);
zarr = [];
for i = 1:length(d)
% modified by Ziji Wu, to accomodate lower case file names. And, it seems
% Matlab cannot tell the difference in file names like T.0.WC and T.0.CT!
%   a = sscanf(d(i).name,'T.%g.WC');
    b = upper(d(i).name);
    if (length(b)<3) || (~strcmp(b(length(b)-2:end),'.WC'))
        continue;
    end
    a = sscanf(b,'T.%g.WC');
  
  if (isempty(a))
    continue;
  end
  zarr = [zarr;a];
end
zarr = sort(zarr);

%% Get contour id's
[pat_id,mov_id,tst_id] = get_ids_old(contournames_fn);

%% Get header info for mha file
%% We'll use the CMS filename for this
z1 = min(diff(zarr));
z2 = max(diff(zarr));
if (z1 ~= z2)
  error ('Sorry, there are uneven slice thicknesses\n');
end

%% Read patient contours
if (~isempty(pat_id))
  id = pat_id;
  for i=1:length(zarr)
    fn = [maskdir '/T.' num2str(zarr(i)) '.WC'];
    PAT(i,:,:) = render_contour(fn,id,x,y);
  end

  P2 = PAT;
  for i=1:size(P2,1)
    P2(i,:,:) = squeeze(PAT(i,:,:))';
  end
  P2 = shiftdim(P2,1);

  writemha(patient_outfn,P2,[ipp(1) ipp(2) min(zarr)],...
	   [ps(1) ps(2) z1],'uchar');
end

%% Read moving contours
if (~isempty(tst_id) || ~isempty(mov_id))
  MOV = zeros(length(zarr),512,512);
  if (~isempty(tst_id))
    id = tst_id;
    for i=1:length(zarr)
      fn = [maskdir '/T.' num2str(zarr(i)) '.WC'];
      MOV(i,:,:) = render_contour(fn,id,x,y);
    end
    MOV = logical(MOV);
  end

  if (sum(sum(sum(double(MOV))))==0)
    id = mov_id;
    MOV = zeros(length(zarr),512,512);
    for i=1:length(zarr)
      fn = [maskdir '/T.' num2str(zarr(i)) '.WC'];
      MOV(i,:,:) = render_contour(fn,id,x,y);
    end
    MOV = logical(MOV);
  end

  M2 = MOV;
  for i=1:size(M2,1)
    M2(i,:,:) = squeeze(MOV(i,:,:))';
  end
  M2 = shiftdim(M2,1);

  writemha(moving_outfn,M2,[ipp(1) ipp(2) min(zarr)],...
	   [ps(1) ps(2) z1],'uchar');

  %% Write non-moving
  NM = P2 & ~M2;
  writemha(nonmoving_outfn,NM,[ipp(1) ipp(2) min(zarr)],...
	   [ps(1) ps(2) z1],'uchar');
end
