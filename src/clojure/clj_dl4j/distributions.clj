(ns clj-dl4j.distributions
  (:import [org.deeplearning4j.nn.conf.distribution Distribution Distributions
                                                    BinomialDistribution ConstantDistribution
                                                    GaussianDistribution LogNormalDistribution
                                                    NormalDistribution OrthogonalDistribution
                                                    TruncatedNormalDistribution UniformDistribution]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                CONSTRUCTORS
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn binomial-distribution
  ^BinomialDistribution
  ([{:keys [n probability]}]
   (binomial-distribution n probability))
  ([n probability]
   (BinomialDistribution. ^int (int n) ^double (double probability))))

(defn constant-distribution
  ^ConstantDistribution
  [spec]
  (let [value (if (map? spec) (:value spec) spec)]
    (ConstantDistribution. ^double (double value))))

(defn gaussian-distribution
  ^GaussianDistribution
  ([{:keys [mean std]}]
   (gaussian-distribution mean std))
  ([mean std]
   (GaussianDistribution. ^double (double mean) ^double (double std))))

(defn log-normal-distribution
  ^LogNormalDistribution
  ([{:keys [mean std]}]
   (log-normal-distribution mean std))
  ([mean std]
   (LogNormalDistribution. ^double (double mean) ^double (double std))))

(defn normal-distribution
  ^NormalDistribution
  ([{:keys [mean std]}]
   (normal-distribution mean std))
  ([mean std]
   (NormalDistribution. ^double (double mean) ^double (double std))))

(defn orthogonal-distribution
  ^OrthogonalDistribution
  [spec]
  (let [gain (if (map? spec) (:gain spec) spec)]
    (OrthogonalDistribution. ^double (double gain))))

(defn truncated-normal-distribution
  ^TruncatedNormalDistribution
  ([{:keys [mean std]}]
   (truncated-normal-distribution mean std))
  ([mean std]
   (TruncatedNormalDistribution. ^double (double mean) ^double (double std))))

(defn uniform-distribution
  ^UniformDistribution
  ([{:keys [lower upper]}]
   (uniform-distribution lower upper))
  ([lower upper]
   (UniformDistribution. ^double (double lower) ^double (double upper))))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                PREDICATES
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn binomial-distribution?
  [obj]
  (instance? BinomialDistribution obj))

(defn constant-distribution?
  [obj]
  (instance? ConstantDistribution obj))

(defn gaussian-distribution?
  [obj]
  (instance? GaussianDistribution obj))

(defn log-normal-distribution?
  [obj]
  (instance? LogNormalDistribution obj))

(defn normal-distribution?
  [obj]
  (instance? NormalDistribution obj))

(defn orthogonal-distribution?
  [obj]
  (instance? OrthogonalDistribution obj))

(defn truncated-normal-distribution?
  [obj]
  (instance? TruncatedNormalDistribution obj))

(defn uniform-distribution?
  [obj]
  (instance? UniformDistribution obj))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                            CONDITIONAL COERCION
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

(defn ->binomial-distribution
  ^BinomialDistribution
  [obj]
  (if (binomial-distribution? obj)
    obj
    (binomial-distribution obj)))

(defn ->constant-distribution
  ^ConstantDistribution
  [obj]
  (if (constant-distribution? obj)
    obj
    (constant-distribution obj)))

(defn ->gaussian-distribution
  ^GaussianDistribution
  [obj]
  (if (gaussian-distribution? obj)
    obj
    (gaussian-distribution obj)))

(defn ->log-normal-distribution
  ^LogNormalDistribution
  [obj]
  (if (log-normal-distribution? obj)
    obj
    (log-normal-distribution obj)))

(defn ->normal-distribution
  ^NormalDistribution
  [obj]
  (if (normal-distribution? obj)
    obj
    (normal-distribution obj)))

(defn ->orthogonal-distribution
  ^OrthogonalDistribution
  [obj]
  (if (orthogonal-distribution? obj)
    obj
    (orthogonal-distribution obj)))

(defn ->truncated-normal-distribution
  ^TruncatedNormalDistribution
  [obj]
  (if (truncated-normal-distribution? obj)
    obj
    (truncated-normal-distribution obj)))

(defn ->uniform-distribution
  ^UniformDistribution
  [obj]
  (if (uniform-distribution? obj)
    obj
    (uniform-distribution obj)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                  GENERIC
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;==================================================
;;=======================META=======================
;;==================================================

(def builders
  {:binomial         ->binomial-distribution
   :constant         ->constant-distribution
   :gaussian         ->gaussian-distribution
   :log-normal       ->log-normal-distribution
   :normal           ->normal-distribution
   :orthogonal       ->orthogonal-distribution
   :truncated-normal ->truncated-normal-distribution
   :uniform          ->uniform-distribution})

;;======================================================
;;=======================EXECUTOR=======================
;;======================================================

(defn distribution
  ^Distribution
  ([spec]
   (distribution (:type spec) spec))
  ([key-fn spec]
   ((key-fn builders) spec)))

(defn distribution?
  [obj]
  (instance? Distribution obj))

(defn ->distribution
  ^Distribution
  [obj]
  (if (distribution? obj)
    obj
    (distribution obj)))
