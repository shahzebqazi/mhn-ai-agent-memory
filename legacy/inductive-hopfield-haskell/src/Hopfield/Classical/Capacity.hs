-- Inductive Hopfield Networks — Capacity Measurement
--
-- Utilities for measuring how many random bipolar patterns an N-neuron
-- classical Hopfield network can store and reliably recall.

module Hopfield.Classical.Capacity
    ( measureCapacity
    , recallAccuracy
    , randomPattern
    , randomPatterns
    ) where

import Hopfield.Classical.Types
import Hopfield.Classical.Hebbian (imprint)
import Hopfield.Classical.Update (recallAsync)
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.Random (randomIO)

randomPattern :: Int -> IO Pattern
randomPattern n = V.generateM n (\_ -> do
    b <- randomIO :: IO Bool
    pure (if b then Up else Down))

randomPatterns :: Int -> Int -> IO [Pattern]
randomPatterns n count = sequence (replicate count (randomPattern n))

recallAccuracy :: WeightMatrix -> [Pattern] -> Double
recallAccuracy w patterns =
    let results = map (\p -> recallAsync w p 100 == p) patterns
        correct = length (filter id results)
    in  fromIntegral correct / fromIntegral (length patterns)

measureCapacity :: Int -> Int -> IO Int
measureCapacity n maxPatterns = go 1
  where
    go p
        | p > maxPatterns = pure (p - 1)
        | otherwise = do
            pats <- randomPatterns n p
            let w = imprint n pats
            let acc = recallAccuracy w pats
            if acc < 1.0
                then pure (p - 1)
                else go (p + 1)
