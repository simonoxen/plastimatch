% =========================================================================
% function [track, state, linkLengths] = PropagateParticles(MODEL_PARAM, ...
%     state, curFrame, nClips, tSet, prevLinkLengths)
% 
% Compute weights based on matching cost  + link constaints and do mean 
% field Monte Carlo to propagate samples
% 
% Input:
% 
% Output:
% 
% Author: Rui Li
% Date: 09/20/2009
% =========================================================================
function [track, state, linkLengths] = PropagateParticles(MODEL_PARAM, ...
    state, curFrame, nClips, tSet, prevLinkLengths, nIterations)

% the template set
patterns = tSet.patCache;

% initalize cell arrays
X = cell(nClips, nIterations);
w = cell(nClips,1);

% -------------------------------------------------------------------------
% initalize iteration

% pairwise distance between samples
D = ComputeDistance(state);
for iClip = 1:nClips
    X{iClip, 1} = state{iClip};
    nSamples = size(state{iClip},1);
    % image observation
    w = ComputeObsScore(state{iClip}, patterns);
    % pairwise constraints
    m = CompuateMessage(D, nSamples, iClip, prevLinkLengths);
    % compute the initial sample weights 
    pi_t{iClip, 1} = w .* m;
end


% -------------------------------------------------------------------------
% mean field Monte Carlo
for iIter=2:nIterations
    for iClip = 1:nClips
        % importance sampling
        
        % compuate the massage
        
        % compute the image matching score
        
        % update weights
    end
end

% -------------------------------------------------------------------------
function D = ComputeDistance(state)
% compute the messages in the form of pair wise distance
X = cell2mat(state);
% X now holds all the sampled locations for all the clips
X = X(:, end-1:end);
m = size(X,1);

% D is the distance matrix that holds the pair wise distance between
% all samples, but only the distances of samples from different clips
% will be used, memory is sacrificed for avoiding writing a loop to
% compute the pairwise distances between samples of different clips
[ i j ] = find(triu(ones(m), 1)); % trick to get indices
D = zeros(m, m); % initialise output matrix
D( i + m*(j-1) ) = sqrt(sum(abs( X(i,:) - X(j,:) ).^2, 2));
D( j + m*(i-1) ) = D( i + m*(j-1) );


function w = ComputeObsScore(X_i, patterns)
    nWidths = X_i(:,1);
    % start of widthIdxs
    s = 2; 
    % end of thetaIdxs
    e = 1 + nWidths;
    widthIdxs = X_i(1, s:e);
    
    nThetas = X_i(:, e+1);
    % start of thetaIdxs
    s = e + 2; 
    % end of thetaIdxs
    e = s + nThetas - 1;
    thetaIdxs = X_i(1, s:e);
    
    % W_i is the image observation likelihood for the samples
    % for the iClip
    nLocations = size(X_i,1);
    w = zeros(nWidths*nThetas*nLocations);
    count = 1;
    for wIdx = widthIdxs
        for tIdx = thetaIdxs
            % template
            T = patterns{wIdx, tIdx, 1};
            [rowOffset, colOffset] = patterns(wIdx, tIdx, 3);
            for lIdx = 1:nLocations
                [locRow locCol] = X_i(lIdx, :);
                I = curFrame((locRow-rowOffset):(locRow+rowOffset), ...
                    (locCol-colOffset):(locCol+colOffset));
                % normalize W_i between in the range between 0 to 1
                w(count) = (prcorr2(T, I) + 1) / 2;
                count = count + 1;
            end
        end
    end

function m = CompuateMessage(D, iClip, prevLinkLengths)


% =========================================================================


