-- Inductive Hopfield Networks — Inductive Step
--
-- Bookkeeping for the n -> n+1 growth process. Records what changes
-- when a neuron is added to an existing network.

module Hopfield.Inductive.Step
    ( InductiveRecord(..)
    , growNetwork
    , recordStep
    ) where

import Hopfield.Classical.Types
import Hopfield.Classical.Hebbian (imprint)
import qualified Data.Vector as V

data InductiveRecord = InductiveRecord
    { irPrevN          :: !Int
    , irNewN           :: !Int
    , irPatternsStored :: !Int
    , irPatternsRecall :: !Int
    , irFailureMode    :: !String
    , irNotes          :: !String
    } deriving (Show)

growNetwork :: Network -> Network
growNetwork net =
    let n' = netSize net + 1
    in  mkNetwork n'

recordStep :: Int -> Int -> Int -> String -> String -> InductiveRecord
recordStep prevN patsStored patsRecalled failMode notes = InductiveRecord
    { irPrevN          = prevN
    , irNewN           = prevN + 1
    , irPatternsStored = patsStored
    , irPatternsRecall = patsRecalled
    , irFailureMode    = failMode
    , irNotes          = notes
    }
