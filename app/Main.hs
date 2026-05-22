-- Inductive Hopfield Networks
-- Entry point for running capacity experiments from the command line.
-- Agents: extend this as new experiment types are added.

module Main where

import Hopfield.Classical.Types (Network, mkNetwork)
import Hopfield.Classical.Hebbian (imprint)
import Hopfield.Classical.Update (recallAsync)
import Hopfield.Classical.Capacity (measureCapacity)
import System.IO (hFlush, stdout)

main :: IO ()
main = do
    putStrLn "Inductive Hopfield — Classical Capacity Experiment"
    putStrLn "=================================================="
    putStrLn ""
    runCapacitySweep [2, 3, 4, 5, 6, 8, 10]

runCapacitySweep :: [Int] -> IO ()
runCapacitySweep sizes = mapM_ runOne sizes
  where
    runOne n = do
        putStr $ "N=" ++ show n ++ ": "
        hFlush stdout
        result <- measureCapacity n 100
        putStrLn $ "max reliable recall = " ++ show result ++ " patterns"
