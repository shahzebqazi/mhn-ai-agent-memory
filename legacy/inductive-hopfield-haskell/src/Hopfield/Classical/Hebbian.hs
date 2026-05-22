-- Inductive Hopfield Networks — Hebbian Imprinting
--
-- Implements the outer-product learning rule: W = (1/N) * sum_mu (xi_mu * xi_mu^T)
-- with zero diagonal (no self-connections).

module Hopfield.Classical.Hebbian
    ( imprint
    , imprintOne
    , outerProduct
    ) where

import Hopfield.Classical.Types
import Data.Vector (Vector)
import qualified Data.Vector as V

imprint :: Int -> [Pattern] -> WeightMatrix
imprint n patterns =
    let raw = foldl addMatrix (zeroMatrix n) (map outerProduct patterns)
        scaled = scaleMatrix (1.0 / fromIntegral n) raw
    in  zeroDiagonal scaled

imprintOne :: Int -> Pattern -> WeightMatrix
imprintOne n pat = zeroDiagonal $ scaleMatrix (1.0 / fromIntegral n) (outerProduct pat)

outerProduct :: Pattern -> WeightMatrix
outerProduct pat =
    let ds = patternToDoubles pat
    in  V.map (\xi_i -> V.map (\xi_j -> xi_i * xi_j) ds) ds

zeroMatrix :: Int -> WeightMatrix
zeroMatrix n = V.replicate n (V.replicate n 0.0)

addMatrix :: WeightMatrix -> WeightMatrix -> WeightMatrix
addMatrix = V.zipWith (V.zipWith (+))

scaleMatrix :: Double -> WeightMatrix -> WeightMatrix
scaleMatrix s = V.map (V.map (* s))

zeroDiagonal :: WeightMatrix -> WeightMatrix
zeroDiagonal w = V.imap (\i row -> V.imap (\j val -> if i == j then 0.0 else val) row) w
