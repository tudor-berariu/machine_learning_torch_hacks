--[[
   Tudor Berariu, 2015
   Value Iteration example.

   There is a 2x3 grid space with the following rewards:
   +---+---+---+
   | 0 | 0 | 0 |
   +---+---+---+
   | 0 |-1 | 1 |
   +---+---+---+

   When the agent takes a move there's a 0.8 probability that it
   succeedes, and 0.1 for both being deviated to the left or to the
   right.
--]]

do
   require('torch')

   -- There are six states on a 2x3 grid
   local height = 2
   local width = 3
   local dr = torch.Tensor({-1, 0, 1, 0})
   local dc = torch.Tensor({0, 1, 0, -1})
   local Pact = {[-1] = 0.1, [0] = 0.8, [1] = 0.1}

   local P = torch.zeros(height, width, 4, height, width)
   local R = torch.zeros(height, width)
   R[2][2] = -1
   R[2][3] = 1
   local discount = 1

   for row = 1,height do
      for col = 1,width do
         if R[row][col] == 0 then
            for action = 1,4 do
               local Psa = P[row][col][action]
               for result = -1,1 do
                  -- The action that might happenx
                  real_action = (action + result - 1) % 4 + 1
                  -- The next state
                  local next_row = row + dr[real_action]
                  next_row = math.max(math.min(next_row, height), 1)
                  local next_col = col + dc[real_action]
                  next_col = math.max(math.min(next_col, width), 1)
                  -- Update the probability
                  Psa[next_row][next_col] =
                     Psa[next_row][next_col] + Pact[result]
               end -- for r
            end -- for a
         end -- if
      end -- for col
   end -- for row

   print(P)

   local U = torch.zeros(2, 3)
   repeat
      local Uold = U:clone()
      for row = 1, height do
         for col = 1, width do
            Umax = torch.cmul(Uold, P[row][col][1]):sum()
            for action = 2,4 do
               Ua = torch.cmul(Uold, P[row][col][action]):sum()
               Umax = math.max(Umax, Ua)
            end -- for a
            U[row][col] = R[row][col] + discount * Umax
         end -- col
      end -- row
      local delta = (Uold - U):abs():max()
   until delta < 0.001

   print(U)

   local policy = torch.zeros(2, 3)
   for row = 1, height do
      for col = 1, width do
         Umax = torch.cmul(U, P[row][col][1]):sum()
         best_action = 1
         for action = 2, 4 do
            Ua = torch.cmul(U, P[row][col][action]):sum()
            if Ua > Umax then
               Umax = Ua
               best_action = action
            end
         end -- for a
         policy[row][col] = best_action
      end -- for col
   end -- for row

   print(policy)
end
