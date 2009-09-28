% =========================================================================
% function state = GenerateInitialParticleSet(initialTrack, templateSet, ...
%     nClips, nParticles)
% 
% Author:   Rui Li
% Date:     09/17/2009
% =========================================================================
function state = GenerateInitialParticleSet(initialTrack, templateSet, ...
    nClips, nParticles, options)

patWidths = templateSet.patWidths;
nWidths = length(patWidths);
patThetas = templateSet.patThetas;
nThetas = length(patThetas);

state = cell(nClips, 1);
for iClip = 1:nClips
    % the height of the clips do not change
    % two adjacent lengths for the clips
    iWidth = find(patWidths == initialTrack(iClip, 1));
    wIdxs = (iWidth-1):(iWidth+1);
    wIdxs = wIdxs(wIdxs >= 1 & wIdxs <= nWidths);
    
    % two adjacent rotation angles for the clips
    iTheta = find(patThetas == initialTrack(iClip, 3));
    thetaIdxs = iTheta-1:iTheta+1;
    thetaIdxs = thetaIdxs(thetaIdxs >= 1 & thetaIdxs <= nThetas);
    
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
    
    % we only really sample for the x y locations of the clips
    samples = cell(length(thetaIdxs) * length(wIdxs), 3);
    fields = {'widthIdx', 'thetaIdx', 'paticles'};
    count = 1;
    for iWidth = wIdxs
        for iTheta = thetaIdxs
            samples{count, 1} = iWidth;
            samples{count, 2} = iTheta;
            samples{count, 3} = ...
                mvnrnd(initialTrack(iClip, end-1:end), 2* eye(2), nParticles);
            count = count + 1;
        end
    end 
    
    state{iClip} = cell2struct(samples, fields, 2);
end



