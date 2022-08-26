(ns clj-dl4j.normalization
  (:import [org.deeplearning4j.nn.conf GradientNormalization]))

(def gradient-normalizations
  {:none GradientNormalization/None
   :renormalize-l2-per-layer GradientNormalization/RenormalizeL2PerLayer
   :renormalize-l2-per-param-type GradientNormalization/RenormalizeL2PerParamType
   :clip-element-wise-absolute-value GradientNormalization/ClipElementWiseAbsoluteValue
   :clip-l2-per-layer GradientNormalization/ClipL2PerLayer
   :clip-l2-per-param-type GradientNormalization/ClipL2PerParamType})

(defn gradient-normalization?
  [obj]
  (instance? GradientNormalization obj))

(defn ->gradient-normalization
  ^GradientNormalization
  [obj]
  (if (gradient-normalization? obj)
    obj
    (if-let [impl (get gradient-normalizations (if obj obj :none))]
      impl
      (throw (Exception. (str "GRADIENT NORMALIZATION - Unknown gradient normalization type : " obj))))))
