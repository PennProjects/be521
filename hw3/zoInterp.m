function step_window = zoInterp(x, numInterp)

%% Input
% x = row/column vector of values to interpolate
% numInterp = The number of times each value in x needs to be duplicated
% for

%% Output
% step_window = row vector of numInterp coipes of each value of x

%% Definition
x = reshape(x,[],1); %making x a column vector
step_window = reshape((repmat(x, 1, numInterp))', 1, []);

end