-- Inductive Hopfield Networks — Classical Types
--
-- Core data types for bipolar Hopfield networks. Designed for visibility:
-- small explicit representations so the inductive structure is inspectable.
-- Do NOT replace with matrix libraries until the classical phase is complete.

module Hopfield.Classical.Types
    ( Spin(..)
    , Pattern
    , WeightMatrix
    , Network(..)
    , mkNetwork
    , networkSize
    , flipSpin
    , spinToDouble
    , patternToDoubles
    ) where

import Data.Vector (Vector)
import qualified Data.Vector as V

data Spin = Up | Down
    deriving (Eq, Show, Ord)

flipSpin :: Spin -> Spin
flipSpin Up   = Down
flipSpin Down = Up

spinToDouble :: Spin -> Double
spinToDouble Up   =  1.0
spinToDouble Down = -1.0

patternToDoubles :: Pattern -> Vector Double
patternToDoubles = V.map spinToDouble

type Pattern = Vector Spin

type WeightMatrix = Vector (Vector Double)

data Network = Network
    { netSize    :: !Int
    , netWeights :: !WeightMatrix
    , netState   :: !Pattern
    } deriving (Show)

mkNetwork :: Int -> Network
mkNetwork n = Network
    { netSize    = n
    , netWeights = V.replicate n (V.replicate n 0.0)
    , netState   = V.replicate n Up
    }

networkSize :: Network -> Int
networkSize = netSize
