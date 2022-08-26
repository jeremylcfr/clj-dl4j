(ns clj-dl4j.optimization
  (:import [org.deeplearning4j.nn.api OptimizationAlgorithm]))

(def optimization-algorithms
  {:line-gradient-descent        OptimizationAlgorithm/LINE_GRADIENT_DESCENT
   :conjugate-gradient           OptimizationAlgorithm/CONJUGATE_GRADIENT
   :lbfgs                        OptimizationAlgorithm/LBFGS
   :stochastic-gradient-descent  OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT})

(defn optimization-algorithm?
  [obj]
  (instance? OptimizationAlgorithm obj))

(defn ->optimization-algorithm
  ^OptimizationAlgorithm
  [obj]
  (if (optimization-algorithm? obj)
    obj
    (if-let [impl (get optimization-algorithms obj)]
      impl
      (throw (Exception. (str "OPTIMIZATION ALGORITHM - Unknown optimization algorithm : " obj))))))
