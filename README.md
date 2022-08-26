# clj-dl4j

A clojure wrapper around DL4J, a well-known Java ML library built
upon ND4J (the "Java NumPy") and JavaCPP.

## Note

This library is at early stages of development and is so highly subject to future
breaking changes.
Futhermore, functionnalities are added carefully to avoid oversimplification or a rushed
copied/pasted mess. Indeed, the library tries to follow DL4J beta releasing and so avoid
to implement marginal features which are doomed to refactoring or worse.

## Usage

### Dependencies

This library being under AOT, you might skip the following libraries
explicit inclusions which are not mandatory to test examples.

```clojure
[jeremylcfr/clj-nd4j "0.1.0-SNAPSHOT"]
[jeremylcfr/clj-datavec "0.1.0-SNAPSHOT"]
```

Then, include :

```clojure
[jeremylcfr/clj-dl4j "0.1.0-SNAPSHOT"]
```

You can theoritically use CUDA but I never tested it personnaly. For that, you
just have to include :

```clojure
[org.deeplearning4j/deeplearning4j-cuda-{{CUDA_VERSION}} "1.0.0-beta6"]
```

with `{{CUDA_VERSION}}` from `9.2` to `10.2` included.
In that case, this library works the same (in theory again) with some optional
parameters you can use. See DL4J doc.

### Building a network - the XOR example

This basic example show how to build a neural network
and to test it.
You can test it directly by launching a REPL and typing
the commands under ns alias "xor"

```clojure
;; See https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/xor/XorExample.java

;; Define the XOR problem
;; vectorized training dataset
;; Here we provide the exhaustive binary definition
;; of XOR (exclusive OR)
;; First line represents the input (features), let's say X and Y
;; Last line represents the labels (~ results), here the [0 1] = false and [1 0] = true
;; (the problem and thus the network being symetrical, this could be reverted)
;; Let's say here that the first part of the label is the "percentage" of falseness
(def xor-training-data
  (dataset/->dataset
    [[0 0] [0 1] [1 0] [1 1]]
    [[0 1] [1 0] [1 0] [0 1]]))

;; You could have done that using DataVec which would have
;; vectorized a standard input like {x 0 , y 1 , xor 1}
;; from json/yaml or csv
(def xor-training-data
  (ddataset/read-datasets :jackson
    {:type-hint      :regular
     :dataset-type   :classification
     :use-nd4j?      false
     :schema         [:x :y :xor]
     :batch-size     4
     :label-idx      2
     :max-labels     2}
     "data/xor.json"))

;; Now let's define a simple neural network
;; with a single hidden layer of 4 units
(def xor-network-configuration
  {:mini-batch false
   :seed 123
   :weights-updater {:type :sgd
                     :learning-rate 0.1}
   :bias-init 0
   :layers [{:type                   :dense
             :n-in                   2
             :n-out                  4
             :activation-fn          :sigmoid
             :weight-distribution    {:type   :uniform
                                      :lower  0
                                      :upper  1}}
            {:type                   :output
             :n-in                   4
             :n-out                  2
             :activation-fn          :softmax
             :loss-fn                :negative-log-likehood
             :weight-distribution    {:type   :uniform
                                      :lower  0
                                      :upper  1}}]
   :training-listener {:type           :score-iteration
                       :nb-iterations  100}})

;; First, we can try to output a XOR from the untrained network...
(show!)

;; ...without surprise, unless you were lucky enough to initialize good weights,
;; the result is pretty bad
;; Try another time after a training (of 1000 iterations)
(train!)
(show!)

;; This should be better
;; Repeat it, and try it
(train! 100000)
(show!)

;; You can look the result on "real-life" looking data
(xor/show! [[0.15 0.95] [0.25 0.95] [0.35 0.95] [0.45 0.95]])
```

### Other examples

You can find other examples under the corresponding [folder](src/clojure/clj_dl4j/examples).
These examples are direct adaptations of DL4J [official examples](https://github.com/eclipse/deeplearning4j-examples).
The current examples are :
- MNIST anomaly detection : builds a network to decompose then recompose digits from samples, then plots the best and worst
                            ones using Swing
                            /!\ BEWARE /!\ : launching it for the first time will download MNIST data which is dumped under
                                             C://users/{you}/.deeplearning4j/data/MNIST under Windows and probably in the translated
                                             folder under Linux. Total size is ~50Mo


## License

Copyright © 2018 Jérémy Le Corguillé

Apache License 2.0
