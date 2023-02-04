(ns clj-dl4j.layers.supertypes.lstm
  (:require [clj-dl4j.layers.supertypes.recurrent :as super]
            [clj-nd4j.ml.activation :as activation])
  (:import [org.deeplearning4j.nn.conf.layers AbstractLSTM AbstractLSTM$Builder]
           [org.nd4j.linalg.activations Activation]))

(defn build-with
  ^AbstractLSTM$Builder
  [{:keys [gate-activation-fn forget-bias-init allow-fallback?] :as options} ^AbstractLSTM$Builder builder]
  (cond-> (super/build-with options builder)
          gate-activation-fn (.gateActivationFunction ^AbstractLSTM$Builder ^Activation (activation/->activation-fn gate-activation-fn))
          forget-bias-init (.forgetGateBiasInit ^AbstractLSTM$Builder ^double (double forget-bias-init))
          (some? allow-fallback?) (.helperAllowFallback ^AbstractLSTM$Builder ^boolean allow-fallback?)))
          
          