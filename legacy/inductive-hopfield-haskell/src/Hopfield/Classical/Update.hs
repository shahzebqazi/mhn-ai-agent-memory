-- Inductive Hopfield Networks — Asynchronous Update Rule
--
-- Classical Hopfield update: pick neuron i, compute local field
-- h_i = sum_j W_ij * s_j, set s_i = sign(h_i).
-- Asynchronous: one neuron at a time, random or sequential order.

module Hopfield.Classical.Update
    ( updateNeuron
    , updatePass
    , recallAsync
    , localField
    ) where

import Hopfield.Classical.Types
import Data.Vector (Vector)
import qualified Data.Vector as V

localField :: WeightMatrix -> Pattern -> Int -> Double
localField w state i =
    let row = w V.! i
        ds  = patternToDoubles state
    in  V.sum (V.zipWith (*) row ds)

updateNeuron :: WeightMatrix -> Pattern -> Int -> Pattern
updateNeuron w state i =
    let h = localField w state i
        newSpin = if h >= 0.0 then Up else Down
    in  state V.// [(i, newSpin)]

updatePass :: WeightMatrix -> Pattern -> Pattern
updatePass w state = foldl (updateNeuron w) state [0 .. V.length state - 1]

recallAsync :: WeightMatrix -> Pattern -> Int -> Pattern
recallAsync w probe maxIters = go probe maxIters
  where
    go s 0 = s
    go s n =
        let s' = updatePass w s
        in  if s' == s then s else go s' (n - 1)
