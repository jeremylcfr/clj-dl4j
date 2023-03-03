(ns clj-dl4j.layers.supertypes.base-output
  (:require [clj-dl4j.layers.supertypes.feedforward :as super]
            [clj-nd4j.ml.loss :as loss])
  (:import [org.deeplearning4j.nn.conf.layers BaseOutputLayer BaseOutputLayer$Builder]
           [org.nd4j.linalg.lossfunctions ILossFunction LossFunctions]))

(defn build-with
  ^BaseOutputLayer$Builder
  [{:keys [loss-fn has-bias?] :as options} ^BaseOutputLayer$Builder builder]
  (cond-> (super/build-with options builder)
          loss-fn                (.lossFunction ^BaseOutputLayer$Builder ^ILossFunction (loss/->loss-fn loss-fn))
          (boolean? has-bias?)   (.hasBias ^BaseOutputLayer$Builder ^boolean has-bias?)))
