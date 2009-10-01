% =========================================================================
% function state = GenerateInitialParticleSet(initialTrack, ...
%     patWidths, patThetas, nClips, nParticles)
% 
% Author:   Rui Li
% Date:     09/17/2009
% =========================================================================
function [state, linkLengths] = GenerateInitialParticleSet(initialTrack, ...
    patWidths, patThetas, nClips, nParticles)

% total number of possible template size and orientation
nWidths = length(patWidths);
nThetas = length(patThetas);

% linkLengths store the pair wise Euclidean distance between all clips
clipPos = initialTrack(:, end-1:end);
% trick to get indices
[i,j] = find(triu(ones(nClips), 1));
% initialize output matrix
linkLengths = zeros(nClips, nClips);
linkLengths( i + nClips*(j-1) ) = ...
    sqrt(sum(abs( clipPos(i,:) - clipPos(j,:) ).^2, 2));
linkLengths( j + nClips*(i-1) ) = linkLengths( i + nClips*(j-1) );

state = cell(nClips, 1);
for iClip = 1:nClips
    % the height of the clips do not change
    % two adjacent lengths for the clips
    iWidth = find(patWidths == initialTrack(iClip, 1));
    wIdxs = (iWidth-1):(iWidth+1);
    wIdxs = wIdxs(wIdxs >= 1 & wIdxs <= nWidths);
    nWidthIdxs = length(wIdxs);
    
    % two adjacent rotation angles for the clips
    iTheta = find(patThetas == initialTrack(iClip, 3));
    thetaIdxs = iTheta-1:iTheta+1;
    thetaIdxs = thetaIdxs(thetaIdxs >= 1 & thetaIdxs <= nThetas);
    nThetaIdxs = length(thetaIdxs);
    
    % nParticles: number of particles per configuration (width, theta);
    particles = mvnrnd(initialTrack(iClip, end-1:end), eye(2), nParticles);
    particles = unique(int16(particles), 'rows');
    nParticles = size(particles, 1);
    
    % samples, each row stores [#width samples, width indexes,
    %                           # of theta, theta indexes,
    %                           clip postion Y, clip position X]
    samples = [nWidthIdxs(ones(1,nParticles),:), ...
        wIdxs(ones(1,nParticles),:), ...
        nThetaIdxs(ones(1,nParticles),:), ...
        thetaIdxs(ones(1,nParticles),:), ...
        particles(:,1), particles(:,2)];
    

    state{iClip} = samples;
end


%     % generate all subscripts for widths and thetas that indexed
%     % template cache set
%     % the following is equivalent to
%     % wSub = repmat(wIdxs, length(thetaIdxs), 1);
%     wSub = wIdxs(ones(1,length(thetaIdxs)), :);
%     
%     % the following is equivalent to
%     % tSub = repmat(thetaIdxs', 1, length(wIdxs));
%     temp = thetaIdxs(:);
%     tSub = temp(:, ones(1,length(wIdxs)));
% 
%     % the number of all pairings of thetas and widths
%     nWidthThetas =  length(wSub(:));
%     
%     % nParticles: number of particles per configuration (width, theta);
%     particles = mvnrnd(initialTrack(iClip, end-1:end), 2*eye(2), nParticles);
%     particles = unique(int16(particles), 'rows');
%     nParticles = size(particles, 1);
%     
%     % enumerate all 4-tuples of 
%     % [widthIdx, thetaIdx, rowIdx of clip center, colIdx of clip center]    
%     nSamples = nWidthThetas*nParticles;
%     samples = zeros(nSamples, 4);
%     wSub = wSub(:); wSub = wSub(:, ones(1, nParticles)).'; wSub = wSub(:);
%     tSub = tSub(:); tSub = tSub(:, ones(1, nParticles)).'; tSub = tSub(:);
%     samples(:, 1:2) = [wSub tSub];
%     
%     % i, j are the intermediate indices used to do efficient repmat
%     % for matrix particles
%     i = (1:nParticles).'; i = i(:, ones(1,nWidthThetas));
%     j = (1:2).'; j = j(:, 1);
%     samples(:, 3:4) = particles(i, j);
    


