
%% indir = 'f:/4dct/ct/gouthro';
%% outdir = 'g:/reality/4dct/anon-gcs-0005/masks';

%% indir = 'f:/4dct/ct/scansaroli';
%% outdir = 'g:/reality/4dct/anon-gcs-0006/masks';

%% indir = 'g:/reality/4dct-raw/ct/bresnahan-6061';
%% outdir = 'g:/reality/4dct/anon-gcs-0009/masks';

%% indir = 'd:/4dct-raw/ct/jones-7809';
%% outdir = 'd:/4dct-processed/anon-dr-0011/masks';

indir = 'g:/reality/raw/ct/anon-2563'
outdir = 'g:/reality/processed/lung-0025/masks';

% added by Ziji Wu, Jan 2006 to create the output directory automatically
if (~exist(outdir))
    mkdir(outdir);
end
% added by Ziji Wu, Jan 2006 to check the existence of the input directory
% and its naming convention
maskdir0 = [indir '/seg-contours'];
if (~exist(maskdir0))
  maskdir0 = [indir '/cms'];
end
if (~exist(maskdir0))
    disp('ERROR!! There is no cms directory in the RAW image folder!');
    disp('Cannot continue! Quit!!');
    return;
end

nam_conv = 1;
phases = [0:9];
maskdir = [maskdir0 '/phase00'];
if (~exist(maskdir))
  nam_conv = 2;
  maskdir = [maskdir0 '/t0'];
  phases = [0:9];
end
if (~exist(maskdir))
  nam_conv = 3;
  maskdir = [maskdir0 '/dp10'];
  phases = [10:2:28];
end
if (~exist(maskdir))
    disp('ERROR!! Cannot understand the cms subfolder naming convention!');
    disp('Cannot continue! Quit!!');
    return;
end

for pno=phases

  switch nam_conv
   case 1
    maskdir = [maskdir0 '/phase' num2str(pno)];
   case 2
    maskdir = [maskdir0 '/t' num2str(pno)];
   case 3
    maskdir = [maskdir0 '/dp' num2str(pno)];
  end

  dicomdir = [indir, '/t', num2str(pno)];
  if (~exist(dicomdir))
    dicomdir = '';
    if (length(dir([maskdir, '/*.CT'])) > 0)
      cmsdir = maskdir;
    end
  end
  
  % in case there are some phases not outlined, we can still proceed
  if (~exist(maskdir))
    continue;
  end

  moving_outfn = [outdir '/t' num2str(pno) '_moving.mha'];
  nonmoving_outfn = [outdir '/t' num2str(pno) '_nonmoving.mha'];
  patient_outfn = [outdir '/t' num2str(pno) '_patient.mha'];
  disp(sprintf('Rendering phase %d',pno));
  render_cms_dir

end
