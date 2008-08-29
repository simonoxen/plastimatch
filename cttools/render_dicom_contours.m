% This was written to handle the PMH data.  YMMV....

%% dicom_contour_file = 'g:/reality/raw/from-pmh/Lung/ROIs/RTS00025';
%% dicom_image_file = 'g:/reality/raw/from-pmh/Lung/Exhale DICOM/CT03087';
%% mhaout = 'g:/reality/processed/lung-0024/masks/t0_patient.mha';
%% mhaout = 'g:/reality/processed/lung-0024/masks/t5_patient.mha';
%% offset = [-250 -250 -157.5];
%% num_slices = 152;
%% exhale_patient_item = '3';
%% inhale_patient_item = '7';

dicom_contour_file = 'g:/reality/raw/from-pmh/Liver/LiverRoiDicom/RTS00026';
dicom_image_file = 'G:/reality/raw/from-pmh/Liver/LiverInhaleDicom/CT03359';
inhale_mhaout = 'g:/reality/processed/liver-0027/masks/t0_patient.mha';
exhale_mhaout = 'g:/reality/processed/liver-0027/masks/t5_patient.mha';
exhale_patient_item = '16';
inhale_patient_item = '17';
offset = [-250 -250 -147.5];   %% From mha file header
num_slices = 120;              %% From mha file header

disp('Pre-loading');
a = dicominfo(dicom_contour_file);
b = dicominfo(dicom_image_file);

ps = b.PixelSpacing;
ipp = b.ImagePositionPatient;
cr = double([b.Columns,b.Rows]);
x = ipp(1)+ps(1)/2:ps(1):ipp(1)+ps(1)*cr(1);
y = ipp(2)+ps(2)/2:ps(2):ipp(2)+ps(2)*cr(2);
spacing = [ps(1),ps(2),b.SliceThickness];

% item = inhale_patient_item;
% mhaout = inhale_mhaout;
item = exhale_patient_item;
mhaout = exhale_mhaout;

eval(['cs = a.ROIContourSequence.Item_', item, '.ContourSequence;']);

disp('Loading');
for i=1:length(fieldnames(cs))
  fieldname = sprintf('Item_%d',i);
  ci = getfield(cs,fieldname);
  ctd = ci.ContourData;
  ctdr{i} = reshape(ctd,3,length(ctd)/3);
  zarr(i) = ctdr{i}(3,1);
end
[zvals,junk,zidx] = unique(zarr);

%% First, tried with -y & no transpose
%% Next, tried with -y & transpose
%% Third, tried with -x & transpose
%% Fourth, tried with -x, -y, & transpose
%% Fifth tried with transpose
disp('Rendering');
PAT = zeros(512,512,num_slices);
for i=1:length(zarr)
  [J,BW1] = roifill(x,y,zeros(512,512),ctdr{i}(1,:)',ctdr{i}(2,:)');
  PAT(:,:,zidx(i)) = xor(squeeze(PAT(:,:,zidx(i))),BW1');
end
% PAT = 1000 * PAT - 1000;

disp('Writing');
writemha(mhaout,PAT,offset,spacing,'uchar');
