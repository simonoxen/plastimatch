% script to run ClipTracker
path = '\\physics.mgh.harvard.edu\IGRT_Research\fluoro\0001\0000a\';
fileExt = '*.raw';
imageSize = [2048, 1536];
pixelType = 'ushort';
startFrame = 2;
nFrames = 40;
model = 'MI';
templateType = 'cylinder';
minI = 0;
maxI = 2600;
imageROI = [800 550 310 390];
nClips = 3;
nParticles = 30;
ClipTracker(path, fileExt, imageSize, pixelType, startFrame, nFrames, ...
    model, templateType, minI, maxI, imageROI, nClips);