% =========================================================================
% function trackResult = ClipTracker(path, fileExt, startFrame, nFrames, model, ...
%                           templateType)
% 
% Top level function for runnning clip tracker
% 
% Input:
%   path:           the path to the image files
%   fileExt:        file name extension
%   startFrame:     the index to the frame that we want start tracking
%   nFrames:        total number of frames to be tracked
%   model:          which tracker model to use. 
%                   1. multiple independent tracker (MI)
%                   2. mean field Monte Carlo (MFMC)
%                   3. dynamic programming (DP)
%   templateType:  indicate the type of the template which determines 
%                   the shape of the template
% 
% Return:
%   trackResult: the location of each tracked seeds over nFrames
%                matrix size: nFrames x 3
%                each row stores: seedIndex seedX seedY
% 
% Author: Rui Li 
% Date: 09/02/2009
% =========================================================================
function trackResult = ClipTracker(path, fileExt, imageSize, pixelType,...
    startFrame, nFrames, model, templateType, minI, maxI, imageROI, nClips, ...
    nParticles)

% check for minimum number of inputs
if nargin < 1
    error('The ClipTracker function needs to know the path to the images');
end

% file extension
if nargin < 2
    fileExt = '*.raw';
end

% image size
if nargin < 3
    imageSize = [2048, 1536];
end

% image pixel type
if nargin < 4
    pixelType = 'ushort';
end

% check for minimum number of inputs
if nargin < 5
    startFrame = 1;
end

imageFiles = dir([path, fileExt]);

% the default is to track all the images in the directory
if nargin < 6
    nFrames = length(imageFiles);
end

% set default trackermodel if the model is not specified in the input
% arguments
if nargin < 7
    model = 'MI';
end

% set default template type if there is no input type
if nargin < 8
    templateType = 'cylinder';
end

% display range value, equivalent to the win/level values in
% viva
if nargin < 9 
    minI = 0;
    maxI = 2600;
end
if nargin < 10
    maxI = 2600;
end

% choose a region of interest (ROI)
if nargin < 11
    imageROI = [800 550 310 390];
end

% default number of clips to track
if nargin < 12
    nClips = 3;
end

% default number samples per clip per configuration
if nargin < 13
    nParticles = 30;
end

% make template based on the input template type
if strcmpi(templateType, 'cylinder') == 1
    template = InitCylinderTemplate();
elseif strcmpi(templateType, 'circle') == 1
    template = InitCircleTemplate();
end

% read in the first frame and locate the markers
fid = fopen([path, imageFiles(startFrame).name], 'rb');
curFrame = reshape(fread(fid, pixelType), imageSize);
fclose(fid);
curFrame = curFrame';


% initialize the clip tracks
clipTracks = cell(nFrames, 1);
clipTracks{1} = InitalizeTracks(imcrop(curFrame, imageROI), ...
    imageROI, template, nClips);

figure(1);
imshow(curFrame, [minI, maxI]);
DrawTrackedTemplate(nClips, clipTracks{1});


% generate hypothesis for each clip based on the initial estimate
% states is a cell array indexed by the number of clips
% compute the initial linkLength at this step
linkLengths = zeros(nClips, nClips, nFrames);
[state, linkLengths(:,:,1)] = ...
    GenerateInitialParticleSet(clipTracks{1}, template.tSet.patWidths, ...
    template.tSet.patThetas, nClips, nParticles); 

keyboard;

% the index for the last frame
lastFrame = startFrame + nFrames - 1;

% start tracking
% the first 30 frames are essentially tracked by template
% match
count = 2;

for iFrame=startFrame+1:lastFrame
    % accumulate 30 frames, then start learning a weak
    % dynamic model
    if count > 30 && isempty(MODEL_PARAM.P)
        [MODEL_PARAM.P, MODEL_PARAM.LINK_MEAN, MODEL_PARAM.LINK_VAR] = ...
            LearnModelParams(clipTracks, linkLengths);
    end
    % open the current frame and display it
    fid = fopen([path, imageFiles(iFrame).name], 'rb');
    curFrame = reshape(fread(fid, pixelType), imageSize);
    curFrame = curFrame';    
    
    [clipTracks{count}, state, linkLengths{count}] = ...
        PropagateParticles(MODEL_PARAM, state, curFrame, nClips, ...
        template.tSet, linkLengths{count-1});
    
    % show current tracks
    clf;
    imshow(curFrame, [minI, maxI]);
    DrawTrackedTemplate(nClips, clipTracks{count});
    drawnow; pause(0.01);
    
    count = count + 1;
end

