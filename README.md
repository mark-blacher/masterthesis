# Design and implementation of vectorized sorting algorithms
This repository contains my **master thesis** from 2018 on the design and implementation of vectorized sorting algorithms.

In the master thesis I investigate the **suitability of vector instructions** for common sorting algorithms like **sorting networks**, **mergesort** and **quicksort**.

The master thesis is written **in German**. But it deals with the topic of vectorized sorting in **more detail than** for example a **paper** can do, so it is here for your interest including the original source code.

## Some side notes

* This **master thesis resulted in a paper** [1], in which we essentially present the **vectorized quicksort** from the master thesis.
* Frank Thiemicke, a former student of mine wrote a **bachelor thesis** [2] on vectorizing quicksort with AVX-512 instructions. 
* Jan Wassenberg from Google Research, with whom we collaborated, implemented an **instruction-set-agnostic version of vectorized quicksort** [3] using the highway library [4].
* There is also a **vectorized quicksort** version for **AVX-512** from **Intel** [5], which also uses some of the concepts originally presented in the master thesis.

## References

* [1] Fast and Robust Vectorized In-Place Sorting of Primitive Types
    https://drops.dagstuhl.de/opus/volltexte/2021/13775/
    
* [2] Implementation of a vectorized Quicksort using AVX-512 intrinsics
    https://elib.dlr.de/145402/
    
* [3] Vectorized and performance-portable quicksort https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3142

* [4] https://github.com/google/highway

* [5] https://github.com/intel/x86-simd-sort
