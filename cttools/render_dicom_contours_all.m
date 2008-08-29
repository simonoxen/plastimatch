% this code is to convert DICOM format structure contours to segmented MHA
% volumes. Based on render_dicom_contours.m
%
% Usage: render_dicom_contours_all(dicom_contour_fn, offset, spacing,
% image_volume_size, MHA_folder);
%    dicom_countour_fn - name of the DICOM file that has structure
%       contours.
%    offset, spacing, image_volume_size - offset, voxel spacing, and
%       dimensions in voxels of the image volume the contours were drawn on
%    MHA_folder - the name of the output folder. The code will write MHA
%       volumes with structure name as the file names to the directory.
%
% written by Ziji Wu, 4/6/7

function [] = render_dicom_contours_all(dicom_contour_fn, offset, spacing, sz, MHA_folder)

% dicom_contour_file = 'D:\Raw\CT\new_data\0037\103\1.2.840.113619.2.55.1.1762853477.2038.1159452557.168.103_0103_000295_1170203303646a-no-phi.v2';
dicom_contour_fn = 'G:/reality/new-data/0045/103/1.2.840.113619.2.55.1.1762853477.1992.1161953262.296.103_0103_000765_1170279506ba95-no-phi.v2';
offset = [-244 -244 -197.5];
spacing = [0.953125 0.953125 2.5];
sz = [512 512 148];
MHA_folder = 'G:/reality/processed-3.4.0/0045/masks';

info = dicominfo(dicom_contour_fn);

x = offset(1)+spacing(1)/2:spacing(1):offset(1)+spacing(1)*sz(1);
y = offset(2)+spacing(2)/2:spacing(2):offset(2)+spacing(2)*sz(2);

disp(['There are totaling ' num2str(length(fieldnames(info.StructureSetROISequence))) ' structures to be processed.']);
% loop over each structure
for j=1:length(fieldnames(info.StructureSetROISequence))
    fieldname1 = ['Item_' num2str(j)];
    ROIName = getfield(getfield(info.StructureSetROISequence, fieldname1), 'ROIName');
    disp(['Processing structure ' ROIName '...']);
    
    eval(['cs = info.ROIContourSequence.', fieldname1, '.ContourSequence;']);
    for i=1:length(fieldnames(cs))
        fieldname2 = ['Item_' num2str(i)];
        ci = getfield(cs,fieldname2);
        ctd = ci.ContourData;
        ctdr{i} = reshape(ctd,3,length(ctd)/3);
        zarr(i) = ctdr{i}(3,1);
    end
% why do this? (ZW)     [zvals,junk,zidx] = unique(zarr);
    z = round((zarr-offset(3))/spacing(3));
%     z0 = (zarr-offset(3))/spacing(3);
%     t = find(abs(z-z0)>1.e-3);
%     if (length(t)>0)
%         disp('   z index is not integer!!!');
%     end

    PAT = zeros(sz);
    for i=1:length(zarr)
        [junk,BW1] = roifill(x,y,zeros(sz(1), sz(2)),ctdr{i}(1,:)',ctdr{i}(2,:)');
% ZW don't think this is correct because this will always put the ROI at the bottom of the volume
%        PAT(:,:,zidx(i)) = xor(squeeze(PAT(:,:,zidx(i))),BW1');
        PAT(:,:,z(i)) = xor(squeeze(PAT(:,:,z(i))),BW1');
    end

    disp('Writing');
%    writemha([MHA_folder '\' ROIName '.mha'],int16(PAT),offset,spacing,'short');
    if (~exist(MHA_folder,'dir'))
      mkdir (MHA_folder);
    end
    if (~exist(MHA_folder,'dir'))
      error(sprintf('Could not create directory %s',MHA_folder));
    end
    writemha([MHA_folder '\' ROIName '.mha'],PAT,offset,spacing,'float');
    clear fieldname1 ROIName cs ci ctd ctdr zarr fieldname2 junk PAT BW1 
end
