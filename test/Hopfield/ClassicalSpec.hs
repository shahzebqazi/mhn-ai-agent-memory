module Hopfield.ClassicalSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Hopfield.Classical.Types
import Hopfield.Classical.Hebbian
import Hopfield.Classical.Update
import Hopfield.Classical.Energy
import qualified Data.Vector as V

mkPattern :: [Spin] -> Pattern
mkPattern = V.fromList

spec :: Spec
spec = do
    describe "n=2 classical Hopfield" $ do

        it "stores and recalls a single pattern [Up, Down]" $ do
            let pat = mkPattern [Up, Down]
                w   = imprint 2 [pat]
                out = recallAsync w pat 10
            out `shouldBe` pat

        it "stores and recalls a single pattern [Down, Up]" $ do
            let pat = mkPattern [Down, Up]
                w   = imprint 2 [pat]
                out = recallAsync w pat 10
            out `shouldBe` pat

        it "recalls complement of stored pattern (attractor symmetry)" $ do
            let pat  = mkPattern [Up, Down]
                comp = mkPattern [Down, Up]
                w    = imprint 2 [pat]
                out  = recallAsync w comp 10
            out `shouldBe` comp

        it "has zero diagonal in weight matrix" $ do
            let pat = mkPattern [Up, Down]
                w   = imprint 2 [pat]
            (w V.! 0) V.! 0 `shouldBe` 0.0
            (w V.! 1) V.! 1 `shouldBe` 0.0

        it "energy decreases or stays constant after update pass" $ do
            let pat   = mkPattern [Up, Down]
                w     = imprint 2 [pat]
                probe = mkPattern [Down, Down]
                e0    = energy w probe
                s1    = updatePass w probe
                e1    = energy w s1
            e1 `shouldSatisfy` (<= e0 + 1e-10)

    describe "n=3 classical Hopfield" $ do

        it "stores and recalls one pattern" $ do
            let pat = mkPattern [Up, Down, Up]
                w   = imprint 3 [pat]
                out = recallAsync w pat 10
            out `shouldBe` pat

        it "energy is negative at stored pattern" $ do
            let pat = mkPattern [Up, Down, Up]
                w   = imprint 3 [pat]
                e   = energy w pat
            e `shouldSatisfy` (< 0)

    describe "Hebbian outer product" $ do

        it "outer product of [1,-1] is [[1,-1],[-1,1]]" $ do
            let pat = mkPattern [Up, Down]
                op  = outerProduct pat
            (op V.! 0) V.! 0 `shouldBe` 1.0
            (op V.! 0) V.! 1 `shouldBe` (-1.0)
            (op V.! 1) V.! 0 `shouldBe` (-1.0)
            (op V.! 1) V.! 1 `shouldBe` 1.0
