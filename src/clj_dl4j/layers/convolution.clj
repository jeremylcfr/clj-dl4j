(ns clj-dl4j.layers.convolution
  (:require [clj-dl4j.layers.supertypes.feedforward :as super]
            [clj-dl4j.convolution :as components]
            [clj-java-commons.core :refer [->int-array]])
  (:import [org.deeplearning4j.nn.conf.layers ConvolutionLayer ConvolutionLayer$Builder Convolution1DLayer Convolution1DLayer$Builder
                                              Convolution3D Convolution3D$Builder Deconvolution2D Deconvolution2D$Builder
                                              DepthwiseConvolution2D DepthwiseConvolution2D$Builder]
           [org.deeplearning4j.nn.conf ConvolutionMode]
           [org.deeplearning4j.nn.conf.layers Convolution3D Convolution3D$DataFormat]))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                   GENERIC
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================BUILDER=======================
;;=====================================================

;;=================Raw===============

;; @TradeOff : Maybe make it compatible with single numbers for 1D, but is it really worth ?
(defn convolution-layer-builder
  ^ConvolutionLayer$Builder
  ([]
   (ConvolutionLayer$Builder.))
  ([{:keys [kernel-size stride padding] :as options}]
   (let [builder (convolution-layer-builder)]
     (cond-> (super/build-with options builder)
             kernel-size                   (.kernelSize ^ConvolutionLayer$Builder ^ints (->int-array kernel-size))
             stride                        (.stride ^ConvolutionLayer$Builder ^ints (->int-array stride))
             padding                       (.padding ^ConvolutionLayer$Builder ^ints (->int-array padding))))))

;;=================Conditional===============

(defn convolution-layer-builder?
  [obj]
  (instance? ConvolutionLayer$Builder obj))

(defn ->convolution-layer-builder
  ^ConvolutionLayer$Builder
  [obj]
  (if (convolution-layer-builder? obj)
    obj
    (convolution-layer-builder obj)))

;;===================================================
;;=======================LAYER=======================
;;===================================================

;;=================Raw===============

(defn convolution-layer
  ^ConvolutionLayer
  [options]
  (.build ^ConvolutionLayer$Builder (->convolution-layer-builder options)))

;;=================Conditional===============

(defn convolution-layer?
  [obj]
  (instance? ConvolutionLayer obj))

(defn ->convolution-layer
  ^ConvolutionLayer
  [obj]
  (if (convolution-layer? obj)
    obj
    (convolution-layer obj)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                              UNIDIMENSIONAL
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================BUILDER=======================
;;=====================================================

;;=================Raw===============

(defn convolution-1d-layer-builder
  ^Convolution1DLayer$Builder
  ([]
   (Convolution1DLayer$Builder.))
  ([{:keys [kernel-size stride padding] :as options}]
   (let [builder (convolution-1d-layer-builder)]
     (cond-> (super/build-with options builder)
             kernel-size                   (.kernelSize ^Convolution1DLayer$Builder ^int (int kernel-size))
             stride                        (.stride ^Convolution1DLayer$Builder ^int (int stride))
             padding                       (.padding ^Convolution1DLayer$Builder ^int (int padding))))))

;;=================Conditional===============

(defn convolution-1d-layer-builder?
  [obj]
  (instance? Convolution1DLayer$Builder obj))

(defn ->convolution-1d-layer-builder
  ^Convolution1DLayer$Builder
  [obj]
  (if (convolution-1d-layer-builder? obj)
    obj
    (convolution-1d-layer-builder obj)))

;;===================================================
;;=======================LAYER=======================
;;===================================================

;;=================Raw===============

(defn convolution-1d-layer
  ^Convolution1DLayer
  [options]
  (.build ^Convolution1DLayer$Builder (->convolution-1d-layer-builder options)))

;;=================Conditional===============

(defn convolution-1d-layer?
  [obj]
  (instance? Convolution1DLayer obj))

(defn ->convolution-1d-layer
  ^Convolution1DLayer
  [obj]
  (if (convolution-1d-layer? obj)
    obj
    (convolution-1d-layer obj)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                                 TRIDIMENSIONAL
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================BUILDER=======================
;;=====================================================

;;=================Raw===============

;; @TradeOff : Maybe make it compatible with single numbers for 1D, but is it really worth ?
(defn convolution-3d-layer-builder
  ^Convolution3D$Builder
  ([]
   (Convolution3D$Builder.))
  ([{:keys [convolution-mode data-format kernel-size stride padding dilation] :as options}]
   (let [builder (convolution-3d-layer-builder)]
     (cond-> (super/build-with options builder)
             convolution-mode              (.convolutionMode ^Convolution3D$Builder ^ConvolutionMode (components/->convolution-mode convolution-mode))
             data-format                   (.dataFormat ^Convolution3D$Builder ^Convolution3D$DataFormat (components/->convolution-data-format data-format))
             kernel-size                   (.kernelSize ^Convolution3D$Builder ^ints (->int-array kernel-size))
             stride                        (.stride ^Convolution3D$Builder ^ints (->int-array stride))
             padding                       (.padding ^Convolution3D$Builder ^ints (->int-array padding))))))

;;=================Conditional===============

(defn convolution-3d-layer-builder?
  [obj]
  (instance? Convolution3D$Builder obj))

(defn ->convolution-3d-layer-builder
  ^Convolution3D$Builder
  [obj]
  (if (convolution-3d-layer-builder? obj)
    obj
    (convolution-3d-layer-builder obj)))

;;===================================================
;;=======================LAYER=======================
;;===================================================

;;=================Raw===============

(defn convolution-3d-layer
  ^Convolution3D
  [options]
  (.build ^Convolution3D$Builder (->convolution-3d-layer-builder options)))

;;=================Conditional===============

(defn convolution-3d-layer?
  [obj]
  (instance? Convolution3D obj))

(defn ->convolution-3d-layer
  ^Convolution3D
  [obj]
  (if (convolution-3d-layer? obj)
    obj
    (convolution-3d-layer obj)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                          BIDIMENSIONAL DECONVOLUTION
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================BUILDER=======================
;;=====================================================

;;=================Raw===============

;; @TradeOff : Maybe make it compatible with single numbers for 1D, but is it really worth ?
(defn deconvolution-2d-layer-builder
  ^Deconvolution2D$Builder
  ([]
   (Deconvolution2D$Builder.))
  ([{:keys [convolution-mode kernel-size stride padding] :as options}]
   (let [builder (deconvolution-2d-layer-builder)]
     (cond-> (super/build-with options builder)
             convolution-mode              (.convolutionMode ^Convolution3D$Builder ^ConvolutionMode (components/->convolution-mode convolution-mode))
             kernel-size                   (.kernelSize ^Deconvolution2D$Builder ^ints (->int-array kernel-size))
             stride                        (.stride ^Deconvolution2D$Builder ^ints (->int-array stride))
             padding                       (.padding ^Deconvolution2D$Builder ^ints (->int-array padding))))))

;;=================Conditional===============

(defn deconvolution-2d-layer-builder?
  [obj]
  (instance? Deconvolution2D$Builder obj))

(defn ->deconvolution-2d-layer-builder
  ^Deconvolution2D$Builder
  [obj]
  (if (deconvolution-2d-layer-builder? obj)
    obj
    (deconvolution-2d-layer-builder obj)))

;;===================================================
;;=======================LAYER=======================
;;===================================================

;;=================Raw===============

(defn deconvolution-2d-layer
  ^Deconvolution2D
  [options]
  (.build ^Deconvolution2D$Builder (->deconvolution-2d-layer-builder options)))

;;=================Conditional===============

(defn deconvolution-2d-layer?
  [obj]
  (instance? Deconvolution2D obj))

(defn ->deconvolution-2d-layer
  ^Deconvolution2D
  [obj]
  (if (deconvolution-2d-layer? obj)
    obj
    (deconvolution-2d-layer obj)))

;;////////////////////////////////////////////////////////////////////////////
;;============================================================================
;;                           BIDIMENSIONAL DEPTH-WISE
;;============================================================================
;;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

;;=====================================================
;;=======================BUILDER=======================
;;=====================================================

;;=================Raw===============

;; @TradeOff : Maybe make it compatible with single numbers for 1D, but is it really worth ?
(defn depth-wise-convolution-2d-layer-builder
  ^DepthwiseConvolution2D$Builder
  ([]
   (DepthwiseConvolution2D$Builder.))
  ([{:keys [kernel-size stride padding depth-multiplier] :as options}]
   (let [builder (depth-wise-convolution-2d-layer-builder)]
     (cond-> (super/build-with options builder)
             kernel-size                   (.kernelSize ^DepthwiseConvolution2D$Builder ^ints (->int-array kernel-size))
             stride                        (.stride ^DepthwiseConvolution2D$Builder ^ints (->int-array stride))
             padding                       (.padding ^DepthwiseConvolution2D$Builder ^ints (->int-array padding))
             depth-multiplier              (.depthMultiplier ^DepthwiseConvolution2D$Builder ^int (int depth-multiplier))))))

;;=================Conditional===============

(defn depth-wise-convolution-2d-layer-builder?
  [obj]
  (instance? DepthwiseConvolution2D$Builder obj))

(defn ->depth-wise-convolution-2d-layer-builder
  ^DepthwiseConvolution2D$Builder
  [obj]
  (if (depth-wise-convolution-2d-layer-builder? obj)
    obj
    (depth-wise-convolution-2d-layer-builder obj)))

;;===================================================
;;=======================LAYER=======================
;;===================================================

;;=================Raw===============

(defn depth-wise-convolution-2d-layer
  ^DepthwiseConvolution2D
  [options]
  (.build ^DepthwiseConvolution2D$Builder (->depth-wise-convolution-2d-layer-builder options)))

;;=================Conditional===============

(defn depth-wise-convolution-2d-layer?
  [obj]
  (instance? DepthwiseConvolution2D obj))

(defn ->depth-wise-convolution-2d-layer
  ^DepthwiseConvolution2D
  [obj]
  (if (depth-wise-convolution-2d-layer? obj)
    obj
    (depth-wise-convolution-2d-layer obj)))

;; Separable needs constraints, see later
