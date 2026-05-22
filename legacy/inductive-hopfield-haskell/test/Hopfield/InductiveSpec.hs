module Hopfield.InductiveSpec (spec) where

import Test.Hspec
import Hopfield.Classical.Types
import Hopfield.Classical.Hebbian
import Hopfield.Classical.Update
import Hopfield.Inductive.Step
import qualified Data.Vector as V

mkPattern :: [Spin] -> Pattern
mkPattern = V.fromList

spec :: Spec
spec = do
    describe "Inductive step n=2 -> n=3" $ do

        it "grown network has size n+1" $ do
            let net  = mkNetwork 2
                net' = growNetwork net
            networkSize net' `shouldBe` 3

        it "pattern stored at n=2 can be embedded in n=3 by padding" $ do
            let pat2 = mkPattern [Up, Down]
                pat3 = V.snoc pat2 Up
                w3   = imprint 3 [pat3]
                out  = recallAsync w3 pat3 10
            out `shouldBe` pat3

    describe "Inductive record" $ do

        it "records a step with correct fields" $ do
            let rec = recordStep 2 1 1 "none" "trivial at n=2"
            irPrevN rec `shouldBe` 2
            irNewN rec `shouldBe` 3
            irPatternsStored rec `shouldBe` 1
            irPatternsRecall rec `shouldBe` 1
            irFailureMode rec `shouldBe` "none"
