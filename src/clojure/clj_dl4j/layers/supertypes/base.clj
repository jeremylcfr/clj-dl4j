(ns clj-dl4j.layers.supertypes.base
  (:require [clj-dl4j.layers.supertypes.layer :as super]
            [clj-dl4j.distributions :as distributions]
            [clj-dl4j.weights :as weights]
            [clj-dl4j.updaters :as updaters]
            [clj-dl4j.normalization :as gnorm]
            [clj-nd4j.ml.activation :as activation])
  (:import [org.deeplearning4j.nn.conf Updater GradientNormalization]
           [org.deeplearning4j.nn.conf.layers BaseLayer BaseLayer$Builder]
           [org.deeplearning4j.nn.conf.distribution Distribution]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf.weightnoise IWeightNoise WeightNoise]
           [org.nd4j.linalg.learning.config IUpdater]
           [org.nd4j.linalg.activations Activation]))

(defn build-with
  ^BaseLayer$Builder
  [{:keys [activation-fn
           weight-init-method weight-distribution l1-weights l2-weights weight-noise weights-updater
           bias-init l1-bias l2-bias bias-updater
           gradient-normalization gradient-normalization-threshold] :as options} ^BaseLayer$Builder builder]
  (cond-> (super/build-with options builder)
          activation-fn                       (.activation ^BaseLayer$Builder ^Activation (activation/->activation-fn activation-fn))
          weight-init-method                  (.weightInit ^BaseLayer$Builder ^WeightInit (weights/->weight-init weight-init-method))
          weight-distribution                 (.weightInit ^BaseLayer$Builder ^Distribution (distributions/->distribution weight-distribution))
          l1-weights                          (.l1 ^BaseLayer$Builder ^double (double l1-weights))
          l2-weights                          (.l2 ^BaseLayer$Builder ^double (double l2-weights))
          weight-noise                        (.weightNoise ^BaseLayer$Builder ^IWeightNoise (weights/->weight-noise weight-noise))
          weights-updater                     (.updater ^BaseLayer$Builder ^IUpdater (updaters/->updater weights-updater))
          bias-init                           (.biasInit ^BaseLayer$Builder ^double (double bias-init))
          l1-bias                             (.l1Bias ^BaseLayer$Builder ^double (double l1-bias))
          l2-bias                             (.l2Bias ^BaseLayer$Builder ^double (double l2-bias))
          bias-updater                        (.biasUpdater ^BaseLayer$Builder ^IUpdater (updaters/->updater bias-updater))
          gradient-normalization              (.gradientNormalization ^BaseLayer$Builder ^GradientNormalization (gnorm/->gradient-normalization gradient-normalization))
          gradient-normalization-threshold    (.gradientNormalizationThreshold ^BaseLayer$Builder ^double (double gradient-normalization-threshold))))
