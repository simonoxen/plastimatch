function ct = get_cms_ct(fn)
%% This used to be called readcms(), and it only returned the image 

fp = fopen(fn,'r','b');   %% Open big-endian
if (fp == -1)
  error ('Cannot open cms file for reading');
end

%% Skip lines
for i=1:9
  fgets(fp,2048);
end

%% Line 10, get resolution
xoff = fscanf(fp,'%g',1);
fscanf(fp,',');
yoff = fscanf(fp,'%g',1);
fgets(fp,2048);

%% Line 11, get resolution
xres = fscanf(fp,'%d',1);
fscanf(fp,',');
yres = fscanf(fp,'%d',1);
fgets(fp,2048);

%% Skip lines
for i=1:3
  fgets(fp,2048);
end

%% Line 15, get slice thickness
fscanf(fp,'%d',1);
fscanf(fp,',');
slice_thickness = fscanf(fp,'%g',1);
fgets(fp,2048);

%% Skip lines
for i=1:3
  fgets(fp,2048);
end

%% Line 19, get z position
zpos = fscanf(fp,'%g',1);
fgets(fp,2048);

%% Skip lines
for i=1:1
  fgets(fp,2048);
end

%% Line 21, get pixel size
xpixsize = fscanf(fp,'%g',1);
fscanf(fp,',');
ypixsize = fscanf(fp,'%g',1);
fgets(fp,2048);

%% Skip to end of header, which is always 1024 bytes
%% (though I suppose it could be written on line 2)
hdr_size = 1024;
fseek(fp,hdr_size,'bof');

[A,count] = fread(fp,[yres,xres],'*int16');
%% A = int16(bitor(bitshift(A,8),bitshift(A,-8)));

A(A<=-32768) = -1000;    %% For GE scanner
fclose(fp);

%% GCS -- sometimes this is necessary, sometimes not
%% A = A';

ct.xoff = -xoff;
ct.yoff = -yoff;
ct.xres = xres;
ct.yres = yres;
ct.zpos = zpos;
ct.slice_thickness = slice_thickness;
ct.xpixsize = xpixsize;
ct.ypixsize = ypixsize;

ct.img = A;
