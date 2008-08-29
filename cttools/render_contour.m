function A = render_contour(fn,id,x,y)
%% A = render_contour(fn,id,x,y)

Header_size = 5;

%% id = 1;

Object_ID = id;
A = [];

%% fclose(fid_input);

fid_input = fopen(fn,'r');

% Skip the header lines
for j=1:Header_size
  tline = fgetl(fid_input);
end

disp(['Rendering file: ',fn]);

%% C = [];
clear C;
cno = 0;
while 1
% Read the number of points
  num_of_points_temp = fscanf(fid_input,'%d',1);
% num_of_points_temp == 0 -> near the end of file (virtually end)
  if(num_of_points_temp == 0)
    break
  end
  contour_num = fscanf(fid_input,'%d',1);
  num_of_line = ceil(num_of_points_temp/5);
  %% If the Object_ID is found, read and record the coordinates
  %% into the temporary file
  if(contour_num == Object_ID)
    disp(['Object ID found - ',num2str(contour_num),' ',num2str(num_of_points_temp)])
    
    tline = fgetl(fid_input);  %% Throw away end of line
    cno = cno + 1;
    C{cno} = [];
    for j=1:num_of_line
      tline = fgetl(fid_input);
      j = sscanf(tline,'%g,');
      j = reshape(j,2,length(j)/2);
      C{cno} = [C{cno},j];
    end
% If the Object_ID is not found, skip the coordinates
  else
    for j=1:num_of_line+1
      junkline = fgetl(fid_input);
    end
  end
end

fclose(fid_input);

%% Render contours into a stack
BW0 = zeros(cno,512,512);
if (cno~=0)
%  figure(1);clf;hold on;
%  styles = 'brgk';
  for c = 1:cno
    disp(sprintf('Filling %d of %d',c,cno));
    [J,BW1] = roifill(x,y,zeros(512,512),C{c}(1,:)',-C{c}(2,:)');
    BW0(c,:,:) = BW1;
%    plot(C{cno}(1,:),-C{cno}(2,:),styles(c));
%    disp('press any key to continue');
%    pause;
  end
%  close;
end

%% Compress stack
A = zeros(512,512);
for c = 1:cno
  is_dup = 0;
  for d = c+1:cno
    dup_thresh = 2;
    tv = sum(sum(xor(squeeze(BW0(c,:,:)),squeeze(BW0(d,:,:)))));
    if (tv <= dup_thresh)
      disp(sprintf('Contour %d detected as duplicate',c));
      is_dup = 1;
      break;
    end
  end
  if (~is_dup)
    A = xor(A,squeeze(BW0(c,:,:)));
  end
end

%% dsp(A,1);
%% pause;
%% close;

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
info = dicominfo(image_fn);
I = dicomread(info);
ps = info.PixelSpacing;
ipp = info.ImagePositionPatient;
cr = [info.Columns,info.Rows];

% x = ipp(1):ps(1):ipp(1)+ps(1)*cr(1);
% y = ipp(2):ps(2):ipp(2)+ps(2)*cr(2);

x = ipp(1)+ps(1)/2:ps(1):ipp(1)+ps(1)*cr(1);
y = ipp(2)+ps(2)/2:ps(2):ipp(2)+ps(2)*cr(2);

% image(x,-y,imnorm(I));
% colormap(gray(256));
% hold on;
% plot(C(1,:),C(2,:),'r');

image(x,y,imnorm(I));
colormap(gray(256));
hold on;
plot(C(1,:),-C(2,:),'r');

x1 = ones(512,1) * x;
y1 = y' * ones(1,512);
x1 = x1(:);
y1 = y1(:);


return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [] = extract_contour_mgh(File_dir, File_extension, Object_ID, Output_file_name)

Header_size = 5;

file_list = dir([File_dir,'*.',File_extension]);
num_of_files = length(file_list);

for i=1:num_of_files
    file_name = file_list(i).name;
    z_val(i) = str2num(file_name(3:length(file_name)-3));
end
z_val_sorted = sort(z_val);

num_of_slices = 0;

fid_temp = fopen('point_temp.txt','w');
for i=1:num_of_files
    % If there is '.5'
    if mod(z_val_sorted(i),1) == 0
        val_string = num2str(z_val_sorted(i),'%.0f');
    else
        val_string = num2str(z_val_sorted(i),'%.1f');
    end

    fid_input = fopen([File_dir,'T.',val_string,'.',File_extension],'r');

    if fid_input == -1
        disp('Error in opening file')
        break
    else
        % Skip the header lines
        for j=1:Header_size
            tline = fgetl(fid_input);
        end

        while 1
            % Read the number of points
            num_of_points_temp = fscanf(fid_input,'%d',1);
            % num_of_points_temp == 0 -> near the end of file (virtually end)
            if(num_of_points_temp == 0)
                break
            end
            contour_num = fscanf(fid_input,'%d',1);
            num_of_line = ceil(num_of_points_temp/5);
            % If the Object_ID is found, read and record the coordinates
            % into the temporary file
            if(contour_num == Object_ID)
                disp(['Object ID found - ',num2str(contour_num)])
                num_of_slices = num_of_slices + 1;
                Num_of_points(num_of_slices) = num_of_points_temp;
                z_cor(num_of_slices) = z_val_sorted(i);

                for j=1:num_of_line+1
                    tline = fgetl(fid_input);
                    for k=1:length(tline)
                        if(tline(k) ~= ',')
                            fprintf(fid_temp,'%c',tline(k));
                        end
                    end
                    fprintf(fid_temp,'\n');
                end
            % If the Object_ID is not found, skip the coordinates
            else
                for j=1:num_of_line+1
                    tline = fgetl(fid_input);
                end
            end
        end
        fclose(fid_input);
    end
end
fclose(fid_temp);

if num_of_slices == 0
    disp('Error - slice not found')
end

point = zeros(sum(Num_of_points),3);
fid_temp = fopen('point_temp.txt','r');
k = 1;
for i=1:num_of_slices
    for j=1:Num_of_points(i)
        point(k,1) = fscanf(fid_temp,'%f',1);
        point(k,2) = fscanf(fid_temp,'%f',1);
        point(k,3) = z_cor(i);
        k = k+1;
    end
end
fclose(fid_temp);
eval('delete point_temp.txt')

% Write the coordinates to the output file
fid_output = fopen(Output_file_name,'w');

fprintf(fid_output,'Iso center: 0 0 0\n');
fprintf(fid_output,'Num of slices: %d\n',num_of_slices);
k = 1;
for i=1:num_of_slices
    fprintf(fid_output,'Num of points: %d\n',Num_of_points(i));
    for j=1:Num_of_points(i)
        fprintf(fid_output,'%f %f %f\n',point(k,1),point(k,2),point(k,3));
        k = k+1;
    end
end

fclose(fid_output);
