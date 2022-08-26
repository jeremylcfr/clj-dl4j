(ns clj-dl4j.weights
  (:require [clj-dl4j.distributions :as distributions])
  (:import [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf.weightnoise IWeightNoise WeightNoise]))

(def weight-inits
  {:distribution WeightInit/DISTRIBUTION
   :zero WeightInit/ZERO
   :ones WeightInit/ONES
   :sigmoid-uniform WeightInit/SIGMOID_UNIFORM
   :normal WeightInit/NORMAL
   :lecun-uniform WeightInit/LECUN_UNIFORM
   :lecun-normal WeightInit/LECUN_NORMAL
   :uniform WeightInit/UNIFORM
   :xavier WeightInit/XAVIER
   :xavier-uniform WeightInit/XAVIER_UNIFORM
   :xavier-fan-in WeightInit/XAVIER_FAN_IN
   :xavier-legacy WeightInit/XAVIER_LEGACY
   :relu WeightInit/RELU
   :relu-uniform WeightInit/RELU_UNIFORM
   :identity WeightInit/IDENTITY
   :var-scaling-normal-fan-in WeightInit/VAR_SCALING_NORMAL_FAN_IN
   :var-scaling-normal-fan-out WeightInit/VAR_SCALING_NORMAL_FAN_OUT
   :var-scaling-normal-fan-avg WeightInit/VAR_SCALING_NORMAL_FAN_AVG
   :var-scaling-uniform-fan-in WeightInit/VAR_SCALING_UNIFORM_FAN_IN
   :var-scaling-uniform-fan-out WeightInit/VAR_SCALING_UNIFORM_FAN_OUT
   :var-scaling-uniform-fan-avg WeightInit/VAR_SCALING_UNIFORM_FAN_AVG})

(defn weight-init?
  [obj]
  (instance? WeightInit obj))

(defn ->weight-init
  ^WeightInit
  [obj]
  (if (weight-init? obj)
    obj
    (if-let [impl (get weight-inits obj)]
      impl
      (throw (Exception. (str "WEIGHT INIT - Unknown weight init type : " obj))))))

(defn ->weight-noise-method
  [method]
  (case method
        :additive            true
        :multiplicative      false
        (throw (Exception. (str "WEIGHT NOISE - Weight noise calculation method unknown : " method " ; must be :additive, :multiplicative or nil (~ :additive)")))))

(defn weight-noise-from-map
  ^WeightNoise
  [{:keys [distribution method apply-to-bias?] :or {apply-to-bias? false , method :additive}}]
  (let [distribution (if-let [d distribution]
                       d
                       (throw (Exception. "WEIGHT NOISE - Distribution is mandatory")))]
    (WeightNoise. ^Distribution (distributions/->distribution distribution)  ^boolean apply-to-bias? ^boolean (->weight-noise-method method))))

(defn weight-noise
  ^IWeightNoise
  ([obj]
   (if (map? obj)
     (weight-noise-from-map obj)
     (WeightNoise. ^Distribution (distributions/->distribution obj))))
  ([distribution method]
   (WeightNoise. ^Distribution (distributions/->distribution distribution) ^boolean (->weight-noise-method method)))
  ([distribution method apply-to-bias?]
   (WeightNoise. ^Distribution (distributions/->distribution distribution) ^bolean apply-to-bias? ^boolean (->weight-noise-method method))))

(defn weight-noise?
  [obj]
  (instance? IWeightNoise obj))

(defn ->weight-noise
  ^IWeightNoise
  [obj]
  (if (weight-noise? obj)
    obj
    (weight-noise obj)))

