function [ind, thresh, flip] = find_best_threshold(X, y, p_dist)
% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

% 
% rand('seed', 0);
% 
% % m datapoints in 2-dimensions
% mm = 150;
% X = rand(mm, 2);
% 
% thresh_pos = .6;
% y = [X(:, 1) < thresh_pos & X(:, 2) < thresh_pos];
% y = 2 * y - 1;

[mm, nn] = size(X);
ind = 1;
thresh = 0;
% p_dist = ones(mm, 1);
% p_dist = p_dist / sum(p_dist);

UB = inf;

% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.

[Xsort, idxSort] = sort(X); 
for j=1:1:nn
%     j
    F_vector = X(:,j);
    FeatureSort = F_vector(idxSort(:,j));
    Ysort = y(idxSort(:,j));
    pSort = p_dist(idxSort(:,j));
    obj = transpose(pSort)*((Ysort==-1).*1);
    for i=1:1:mm
        obj = obj + pSort(i)*Ysort(i);
        f = 1-obj;
        now = 0;
        if obj> f
            flip = -1;
            now = f;
        else
            flip = 1;
            now = obj;
        end
        if now < UB
            UB = now;
            thresh = FeatureSort(i);
            ind = j;
        end
    end
%     thresh 
end











