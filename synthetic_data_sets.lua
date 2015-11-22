-- Tudor Berariu, 2015

local module synthetic_data_sets = {}

--[[
  Creates a data set by sampling from a given noisy function.
   @f represents the function (by default is sin(2*pi*x))
   @N represents the number of examples to be generated
   @noise represents a table where noise.g is a function that generates noise or
        noise.std represents the standard deviation for normally distributed
        noise around 0
   @left and @right represent the interval to generate the x values in
   @uniform is a boolean that tells if examples should be equally spaced or
        chosen from a uniform distribution over @left, @right
--]]
function synthetic_data_sets.random_data(params)
  params = params or {}

  -- The default function is sin(2*pi*x)
  local f = params.f or function(x) return torch.sin(x * 2 * math.pi) end
  assert(type(f) == "function")

  -- The default number of points is 10
  local N = params.N or 10
  assert(type(N) == "number" and N > 0)

  -- The number of dimensions
  local D = params.D or 1
  local K = params.K or 1

  -- The default noise is normally distributed around 0 with std deviation 0.2
  local noise, g
  if params.noise then
    noise = params.noise
  else
    g = function() return torch.normal(0.0, params.noise_std or 0.2) end
    if D == 1 then
      noise = g
    else
      noise = function() return torch.Tensor(D):apply(g) end
    end -- if D == 1
  end -- if params. noise
  assert(type(noise) == "function")

  -- The default interval is [0, 1]
  local left, right
  if D == 1 then
    left = params.left or 0.0
    right = params.right or 1.0
    assert(type(left) == "number" and type(right) == "number" and left < right)
  else
    left = params.left or torch.zeros(D)
    right = params.right or torch.ones(D)
    assert(left:nDimension() == 1 and left:size(1) == D)
    assert(right:nDimension() == 1 and right:size(1) == D)
  end

  local X, T
  -- By default the numbers are not uniformly spaced in [left, right]
  if D == 1 then
    if params.uniform then
      X = torch.linspace(left, right, N)
    else
      X = torch.rand(N):mul(right - left):add(left)
    end -- if params.uniform
  else
    if params.uniform then
      -- generate a lattice of equally spaced (on each dimension) points
      local N1 = torch.floor(math.pow(N, 1/D))
      N = math.pow(N1, D)
      X = torch.Tensor(N, D)
      -- compute X based on N1-basis expansion of N
      for n = 0, N-1 do
        local _n = n
        for d = 1, D do
          X[n+1][d] = (_n % torch.pow(N1, d)) / ((N1-1) * torch.pow(N1, d-1))
          X[n+1][d] = X[n+1][d] * (right[d] - left[d]) + left[d]
          _n = _n - (_n % torch.pow(N1, d))
        end -- for d
      end -- for n
    else
      X = torch.rand(N, D)
      for i = 1,N do
        X[i]:cmul(right - left):add(right)
      end -- for i
    end -- if params.uniform
  end -- if D

  -- Shuffle data
  X = X:index(1, torch.randperm(N):long())

  -- Compute target values
  if K == 1 then
    T = torch.Tensor(N)
  else
    T = torch.Tensor(N, K)
  end -- if K == 1

  for n = 1, N do
    T[n] = f(X[n]) + noise()
  end -- for n

  -- Return X and T
  if D == 1 then
    assert(X:nDimension() == 1 and X:size(1) == N)
  else
    assert(X:nDimension() == 2 and X:size(1) == N and X:size(2) == D)
  end -- if D == 1
  if K == 1 then
    assert(T:nDimension() == 1 and T:size(1) == N)
  else
    assert(T:nDimension() == 2 and T:size(1) == N and T:size(2) == K)
  end -- if K == 1

  return X, T
end

return synthetic_data_sets
