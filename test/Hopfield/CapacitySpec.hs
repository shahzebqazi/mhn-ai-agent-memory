module Hopfield.CapacitySpec (spec) where

import Test.Hspec
import Hopfield.Classical.Types
import Hopfield.Classical.Hebbian
import Hopfield.Classical.Update
import Hopfield.Classical.Capacity
import qualified Data.Vector as V

mkPattern :: [Spin] -> Pattern
mkPattern = V.fromList

spec :: Spec
spec = do
    describe "Capacity at n=2" $ do

        it "can store 1 orthogonal pattern" $ do
            let pat = mkPattern [Up, Down]
                w   = imprint 2 [pat]
                acc = recallAccuracy w [pat]
            acc `shouldBe` 1.0

        it "cannot reliably store 2 arbitrary patterns" $ do
            let p1 = mkPattern [Up, Down]
                p2 = mkPattern [Up, Up]
                w  = imprint 2 [p1, p2]
                acc = recallAccuracy w [p1, p2]
            acc `shouldSatisfy` (< 1.0)

    describe "Capacity at n=4" $ do

        it "can store 1 pattern" $ do
            let pat = mkPattern [Up, Down, Up, Down]
                w   = imprint 4 [pat]
                acc = recallAccuracy w [pat]
            acc `shouldBe` 1.0

    describe "Random capacity measurement" $ do

        it "measureCapacity returns a non-negative integer" $ do
            cap <- measureCapacity 4 10
            cap `shouldSatisfy` (>= 0)
