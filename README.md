# CYNICS

Collection of code I wrote as a part of my time supporting the CYNICS group. All code is loose prototype work focused on testing the plausability and shortcomings of different methods. 

## IWT

The files in this folder are related to deepfake detection, image fingerprinting, and to a lesser extent self healing images. The idea at a high level is that when using Integer Wavelet Decomposition on an image/audio file, we can determine frequency bands for different colors. For each layer deeper we go, the more bare bones of a certian element of the image we get. 

The human eye is limited in it's ability to discern differences in colors after a certian point. For the sake of prototyping, this was chosen semi arbitrarily based off what could be easily seen. By rounding values down to zero after decomposition, a relatively large amount of space could be freed to embed data, be it a reduced complexity version of the cover image, a fingerprinting sequence, or any other data. This data is XORed and encoded, before the inverse of the decomposition is performed. 

The data image size remains functionally unaltered. The data is embedded in a rounding error range such that conventional detection methods will fail to notice any change. Even should the values be noted to be anomalous, determining where the encoded values start and end is by design difficult for a third party. 

## MOKU

A set of scripts designed to perform time series analysis for data collection off of a Moku Go, meant to simulate a basic programmable logic controller (PLC). 
