# RLT

<!-- badges: start -->
[![Travis build status](https://travis-ci.com/teazrq/RLT.svg?branch=master)](https://travis-ci.com/teazrq/RLT)
[![CRAN status](https://www.r-pkg.org/badges/version/RLT)](https://CRAN.R-project.org/package=RLT)
<!-- badges: end -->

This is a new version (>= 4.0.0) of the `RLT` package. Versions prior to 4.0.0 are written in `C` (available at [RLT-Archive](https://github.com/teazrq/RLT-Archive)), while newer versions are based on `C++`. This new version will replace the original CRAN package once it is finished. 

The goal of `RLT` is to provide new functionalities of random forest models. This includes embedded model fit learning a better splitting rule; linear combination splits, confidence intervals, and several other new approaches that are currently being developed. 

## Installation

You can install the released version (currently <= 3.2.2) of `RLT` from [CRAN](https://CRAN.R-project.org/package=RLT) with:

``` r
install.packages("RLT")
```

## New features highlight

  * Unbiased variance estimation (regression forest) based on [Xu, Zhu and Shao (2022+)](https://arxiv.org/abs/2202.09008)
  * [to be implemented] Unbiased survival function confidence band estimation based on Formentini, Liang and Zhu (2022+)
  * Reproducibility in parallel tree version with xoshiro256plus random number generator
  * Speed and space improvement from eariler `c` version
  * [to be implemented] Graph random forests
  * [to be implemented] Python API
    


