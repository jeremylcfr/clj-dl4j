(ns clj-dl4j.configuration
  (:require [clj-dl4j.optimization :as optim]
            [clj-dl4j.distributions :as distributions]
            [clj-dl4j.weights :as weights]
            [clj-dl4j.updaters :as updaters]
            [clj-dl4j.dropout :as dropout]
            [clj-dl4j.convolution :as convolution]
            [clj-dl4j.normalization :as gnorm]
            [clj-dl4j.layers :as layers]
            [clj-dl4j.constraints :as constraints]
            [clj-nd4j.ml.activation :as activation])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration
                                       NeuralNetConfiguration$Builder
                                       NeuralNetConfiguration$ListBuilder
                                       MultiLayerConfiguration
                                       MultiLayerConfiguration$Builder
                                       CacheMode
                                       WorkspaceMode
                                       Updater
                                       GradientNormalization
                                       ConvolutionMode]
           [org.deeplearning4j.nn.conf.distribution Distribution]
           [org.deeplearning4j.nn.conf.layers Layer ConvolutionLayer ConvolutionLayer$AlgoMode]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf.weightnoise IWeightNoise WeightNoise]
           [org.deeplearning4j.nn.conf.dropout IDropout]
           [org.deeplearning4j.nn.api.layers LayerConstraint]
           [org.nd4j.linalg.learning.config IUpdater]
           [org.nd4j.linalg.activations Activation]))

(defn set-workspace-usage!
  ^NeuralNetConfiguration$Builder
  [^NeuralNetConfiguration$Builder builder workspace-usage]
  (case workspace-usage
        :all (do (.trainingWorkspaceMode ^NeuralNetConfiguration$Builder builder WorkspaceMode/ENABLED) (.inferenceWorkspaceMode ^NeuralNetConfiguration$Builder builder WorkspaceMode/ENABLED))
        :training (do (.trainingWorkspaceMode ^NeuralNetConfiguration$Builder builder WorkspaceMode/ENABLED) (.inferenceWorkspaceMode ^NeuralNetConfiguration$Builder builder WorkspaceMode/NONE))
        :inference (do (.trainingWorkspaceMode ^NeuralNetConfiguration$Builder builder WorkspaceMode/NONE) (.inferenceWorkspaceMode ^NeuralNetConfiguration$Builder builder WorkspaceMode/ENABLED))
        true (set-workspace-usage! builder :all)
        false builder))

(defn cache-mode
  ^CacheMode
  [mode]
  (case mode
        :device   CacheMode/DEVICE
        :host     CacheMode/HOST
        :none     CacheMode/NONE
        false     CacheMode/NONE
        (throw (Exception. (str "Unknown cache mode : " mode)))))

(defn cache-mode?
  [mode]
  (instance? CacheMode mode))

(defn ->cache-mode
  ^CacheMode
  [mode]
  (if (cache-mode? mode)
    mode
    (cache-mode mode)))

(defn set-goal-type!
  ^NeuralNetConfiguration$Builder
  [^NeuralNetConfiguration$Builder builder goal-type]
  (case goal-type
        :minimize     (.minimize ^NeuralNetConfiguration$Builder builder true)
        :maximize     (.minimize ^NeuralNetConfiguration$Builder builder false)
        ;; Aliases, not promoted but they exist for the laziest among us
        :min          (set-goal-type! builder :minimize)
        :max          (set-goal-type! builder :maximize)
        (throw (Exception. (str "Unknown goal type : " goal-type ", should be :minimize, :maximize or nil (~ :minimize)")))))

(def test-schema
  {:mini-batch false
   :workspace-usage :all
   :cache-mode :device
   :goal-type :minimize
   :seed 154213
   :optimization-algorithm :stochastic-gradient-descent
   :activation-fn :softmax
   :weight-init-method :xavier
   :weight-noise {:distribution {:type :binomial
                                 :n 5
                                 :probability 0.5}
                  :method :multiplicative
                  :apply-to-bias? true}
   :weights-updater {:type :adamax
                     :learning-rate 0.05
                     :beta1 0.006
                     :beta2 0.008
                     :epsilon 0.0001}
   :l1-weights 0.05
   :l2-weights 0.08
   :bias-init 0.5
   :l1-bias 0.07
   :l2-bias 0.08
   :bias-updater :rmsprop
   :dropout {:type :gaussian
             :probability 0.01}
   :gradient-normalization :renormalize-l2-per-layer
   :gradient-normalization-thresold 0.1
   :convolution-mode :truncate
   :cuda-algo-mode :fastest})

(defn add-constraints-from-map
  ^NeuralNetConfiguration$Builder
  [^NeuralNetConfiguration$Builder builder {:keys [all weights bias]}]
  (cond-> builder
          (and all (not (empty? all)))             (.constrainAllParameters ^NeuralNetConfiguration$Builder #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;" (constraints/->constraints all))
          (and weights (not (empty? weights)))     (.constrainWeights ^NeuralNetConfiguration$Builder #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;" (constraints/->constraints weights))
          (and bias (not (empty? bias)))           (.constrainBias ^NeuralNetConfiguration$Builder #^"[Lorg.deeplearning4j.nn.api.layers.LayerConstraint;" (constraints/->constraints bias))))

(defn add-constraints-from-seq
  ^NeuralNetConfiguration$Builder
 [^NeuralNetConfiguration$Builder builder constraints]
  (add-constraints-from-map builder
    {:all (filter (fn [{:keys [scope]}] (or (nil? scope) (= :all scope))) constraints)
     :weights (filter (fn [{:keys [scope]}] (= :weights scope)) constraints)
     :bias (filter (fn [{:keys [scope]}] (= :bias scope)) constraints)}))

(defn add-constraints
  ^NeuralNetConfiguration$Builder
  [^NeuralNetConfiguration$Builder builder conf]
  (if (map? conf)
    (add-constraints-from-map builder conf)
    (add-constraints-from-seq builder conf)))

;; Add constraints
;; Make clear which options are override over layers
;; and which one are related to the whole network
(defn single-builder
  ^NeuralNetConfiguration$Builder
  [{:keys [;; Meta
           mini-batch workspace-usage cache-mode goal-type
           seed optimization-agorithm
           ;; Layer, if single layer
           layer
           ;; Only when optimization algorithms are line search types : Line Search SGD, Conjugate Gradient, LBFGS
           line-search-iterations

           ;; Layer-specific vars default values, can be overriden from layers
           activation-fn
           weight-init-method weight-distribution l1-weights l2-weights weight-noise weights-updater
           bias-init l1-bias l2-bias bias-updater
           dropout
           gradient-normalization gradient-normalization-threshold
           convolution-mode
           ;; Cuda only, layer-specific too
           cuda-algo-mode
           ;; Constraints
           constraints]
    :or {workspace-usage :all}}]
  (cond-> (NeuralNetConfiguration$Builder.)
          (boolean? mini-batch)                                          (.miniBatch ^NeuralNetConfiguration$Builder mini-batch)
          (or (boolean? workspace-usage) (keyword? workspace-usage))     (set-workspace-usage! workspace-usage)
          cache-mode                                                     (.cacheMode ^NeuralNetConfiguration$Builder (->cache-mode cache-mode))
          goal-type                                                      (set-goal-type! goal-type)
          seed                                                           (.seed ^NeuralNetConfiguration$Builder ^long (long seed))
          optimization-agorithm                                          (.optimizationAlgo ^NeuralNetConfiguration$Builder (optim/->optimization-algorithm optimization-agorithm))
          layer                                                          (.layer ^NeuralNetConfiguration$ListBuilder ^Layer (layers/->layer layer))
          line-search-iterations                                         (.maxNumLineSearchIterations ^NeuralNetConfiguration$Builder ^int (int line-search-iterations))
          activation-fn                                                  (.activation ^NeuralNetConfiguration$Builder ^Activation (activation/->activation-fn activation-fn))
          weight-init-method                                             (.weightInit ^NeuralNetConfiguration$Builder ^WeightInit (weights/->weight-init weight-init-method))
          weight-distribution                                            (.weightInit ^NeuralNetConfiguration$Builder ^Distribution (distributions/->distribution weight-distribution))
          weight-noise                                                   (.weightNoise ^NeuralNetConfiguration$Builder ^IWeightNoise (weights/->weight-noise weight-noise))
          weights-updater                                                (.updater ^NeuralNetConfiguration$Builder ^IUpdater (updaters/->updater weights-updater))
          l1-weights                                                     (.l1 ^NeuralNetConfiguration$Builder ^double (double l1-weights))
          l2-weights                                                     (.l2 ^NeuralNetConfiguration$Builder ^double (double l2-weights))
          bias-init                                                      (.biasInit ^NeuralNetConfiguration$Builder ^double (double bias-init))
          l1-bias                                                        (.l1Bias ^NeuralNetConfiguration$Builder ^double (double l1-bias))
          l2-bias                                                        (.l2Bias ^NeuralNetConfiguration$Builder ^double (double l2-bias))
          bias-updater                                                   (.biasUpdater ^NeuralNetConfiguration$Builder ^IUpdater (updaters/->updater bias-updater))
          dropout                                                        (.dropOut ^NeuralNetConfiguration$Builder ^IDropout (dropout/->dropout dropout))
          gradient-normalization                                         (.gradientNormalization ^NeuralNetConfiguration$Builder ^GradientNormalization (gnorm/->gradient-normalization gradient-normalization))
          gradient-normalization-threshold                               (.gradientNormalizationThreshold ^NeuralNetConfiguration$Builder ^double (double gradient-normalization-threshold))
          convolution-mode                                               (.convolutionMode ^NeuralNetConfiguration$Builder ^ConvolutionMode (convolution/->convolution-mode convolution-mode))
          cuda-algo-mode                                                 (.cudnnAlgoMode ^NeuralNetConfiguration$Builder ^ConvolutionLayer$AlgoMode (convolution/->cuda-convolution-mode cuda-algo-mode))
          constraints                                                    (add-constraints constraints)))


(defn multi-builder
  ^NeuralNetConfiguration$ListBuilder
  [{:keys [with-backprop? with-pretrain? layers]}
   ^NeuralNetConfiguration$Builder builder]
  (let [list-builder (.list ^NeuralNetConfiguration builder)]
    (reduce-kv
      (fn [acc idx layer]
        (.layer ^NeuralNetConfiguration$ListBuilder acc ^int (int idx) ^Layer (layers/->layer layer)))
      list-builder layers)))

;; Warn on layer and layers
(defn network-configuration-builder
  [{:keys [layers] :as options}]
  (let [multi-layer? (< 1 (count layers))
        options* (if multi-layer?
                   (dissoc options :layer)
                   (-> (assoc options :layer (first layers))
                       (dissoc :layers)))]
    (cond->> (single-builder options*)
             multi-layer?      (multi-builder options*))))

(defn network-configuration-builder?
  [obj]
  (or
    (instance? NeuralNetConfiguration$Builder obj)
    (instance? NeuralNetConfiguration$ListBuilder obj)
    (instance? MultiLayerConfiguration$Builder obj)))

(defn ->network-configuration-builder
  [obj]
  (if (network-configuration-builder? obj)
    obj
    (network-configuration-builder obj)))


(defn- network-configuration*
  [options]
  (let [builder (->network-configuration-builder options)]
    (if (instance? NeuralNetConfiguration$Builder builder)
      {:network-type :single , :configuration (.build ^NeuralNetConfiguration$Builder builder)}
      {:network-type :multi  , :configuration (.build ^MultiLayerConfiguration$Builder builder)})))

(defn network-configuration
  [options]
  (:configuration (network-configuration* options)))
