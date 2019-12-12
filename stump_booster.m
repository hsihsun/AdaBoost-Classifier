function [theta, feature_inds, thresholds, flips] = stump_booster(X, y, T)
% STUMP_BOOSTER Uses boosted decision stumps to train a classifier
%
% [theta, feature_inds, thresholds] = stump_booster(X, y, T)
%  performs T rounds of boosted decision stumps to classify the data X,
%  which is an m-by-n matrix of m training examples in dimension n,
%  to match y.
%
%  The returned parameters are theta, the parameter vector in T dimensions,
%  the feature_inds, which are indices of the features (a T dimensional
%  vector taking values in {1, 2, ..., n}), and thresholds, which are
%  real-valued thresholds. The resulting classifier may be computed on an
%  n-dimensional training example by
%
%   theta' * sign(x(feature_inds) - thresholds).
%
%  The resulting predictions may be computed simultaneously on an
%  n-dimensional dataset, represented as an m-by-n matrix X, by
%
%  sign(X(:, feature_inds) - repmat(thresholds', m, 1)) * theta.
%
%  This is an m-vector of the predicted margins.

[mm, nn] = size(X);
p_dist = ones(mm, 1);
p_dist = p_dist / sum(p_dist);

theta = [];
feature_inds = [];
thresholds = [];
flips = [];

for iter = 1:T
  [ind, thresh, flip] = find_best_threshold(X, y, p_dist);
  feature_inds = [feature_inds; ind];
  thresholds = [thresholds; thresh];
  flips = [flips; flip];
  % ------- You should implement your code here -------- %
  if iter > 1  
      for i = 1:1:mm
          tmp=0;
          for t = 1:1:iter-1
              tmp = tmp+flips(t)*theta(t)*...
                  sign(X(i,feature_inds(t))-thresholds(t));
          end
         p_dist(i)=exp(-1*y(i)*tmp);
      end
      p_dist=p_dist./sum(p_dist(:));
  end
  Wp = 0; Wn = 0;
  
  for i =1:1:mm
      if y(i)*flip*sign(X(i,ind)-thresh) == 1
          Wp = Wp+ p_dist(i);
      else
          Wn = Wn+ p_dist(i);
      end
  end
  if Wp~=0 & Wn~=0
      newest_theta_param = (1/2)*log(Wp/Wn);
  else
      newest_theta_param = 0  % Change this line so that newest_theta_param takes
                           % the optimal weight for the new decision stump.
  end
  % ------- No need to change this part ------- %
  theta = [theta; newest_theta_param];
  
end