-- Inductive Hopfield Networks — Energy Function
--
-- Classical Hopfield energy: E = -0.5 * sum_{i/=j} W_ij * s_i * s_j
-- Energy must decrease (or stay constant) on every asynchronous update.

module Hopfield.Classical.Energy
    ( energy
    , energyChange
    ) where

import Hopfield.Classical.Types
import qualified Data.Vector as V

energy :: WeightMatrix -> Pattern -> Double
energy w state =
    let ds = patternToDoubles state
        n  = V.length ds
        total = sum [ (w V.! i) V.! j * (ds V.! i) * (ds V.! j)
                    | i <- [0 .. n - 1]
                    , j <- [0 .. n - 1]
                    , i /= j
                    ]
    in  -0.5 * total

energyChange :: WeightMatrix -> Pattern -> Int -> Double
energyChange w state i =
    let ds  = patternToDoubles state
        row = w V.! i
        h_i = V.sum (V.zipWith (*) row ds)
        s_i = ds V.! i
    in  -2.0 * h_i * s_i
