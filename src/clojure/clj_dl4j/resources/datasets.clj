(ns clj-dl4j.resources.datasets
  (:import [org.nd4j.linalg.dataset.api.iterator BaseDatasetIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]))

;; TODO : intégrer dans ND4J le constructeur basique pour choix avec kw
;; Peu urgentmais très important pour MEP

(defn mnist-dataset-iterator
  ^MnistDataSetIterator
  ([batch-size num-samples]
   (MnistDataSetIterator. ^int (int batch-size) ^int (int num-samples)))
  ([batch-size num-samples binarize?]
   (MnistDataSetIterator. ^int (int batch-size) ^int (int num-samples) ^boolean (boolean binarize?)))
  ([batch-size num-samples binarize? train? shuffle? seed]
   (MnistDataSetIterator. ^int (int batch-size) ^int (int num-samples) ^boolean (boolean binarize?) ^boolean (boolean train?) ^boolean (boolean shuffle?) ^long (long seed))))


