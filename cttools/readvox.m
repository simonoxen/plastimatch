function [A,Ainfo] = readvox(fn)

%% fn = 'gcs.mha';

fp = fopen(fn,'r');
if (fp == -1)
  error ('Cannot open vox file for reading');
end

%% Parse header
t = fgetl(fp);
if (~strcmp(t,'VOX'))
  error ('Not a VOX file');
end

%% Skip line 2
t = fgetl(fp);

%% Get image resolution
t = fgetl(fp);
[sz,cnt] = sscanf(t,'%d');
if (cnt ~= 3)
  error ('Error parsing vox header');
end

%% Get pixel size
t = fgetl(fp);
[Ainfo.ElementSpacing,cnt] = sscanf(t,'%g');
if (cnt ~= 3)
  error ('Error parsing vox header');
end

%% Skip line 5
t = fgetl(fp);

[A,count] = fread(fp,sz(1)*sz(2)*sz(3),'*short');

fclose(fp);

A = reshape(A,sz(1),sz(2),sz(3));
