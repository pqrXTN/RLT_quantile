#' @title           kernel.RLT
#' 
#' @description     Get random forest induced kernel weight matrix of testing samples 
#'                  or between any two sets of data. This is an experimental feature.
#'                  Use at your own risk.
#'                  
#' @param object    A fitted RLT object.
#' 
#' @param X1        The the first dataset. This calculates an \eqn{n_1 \times n_1} kernel
#'                  matrix of `X1`. 
#' 
#' @param X2        The the second dataset of the relative kernel weights are required. 
#'                  If \code{X2} is supplied, then calculate an \eqn{n_1 \times n_2} 
#'                  kernel matrix. If \code{vs.train} is used, then this must be the original 
#'                  training data.
#' 
#' @param vs.train  To calculate the kernel weights with respect to the training data. 
#'                  This is slightly different than supplying the training data to \code{X2}
#'                  due to re-samplings of the training process. Hence, \code{ObsTrack} must
#'                  be available from the fitted object (using \code{resample.track = TRUE}).
#'                  
#' @param OOB       If \code{TRUE}, use OOB trees to estimate kernel and normalize the result,
#'                  and force \code{vs.train = FALSE}, since sample A cannot be both OOB and InB. 
#'                  1)  When \code{X2} is \code{null}, assume \code{X1} are all training data.
#'                  return (kernel / OOB_count) + I. \code{OOB_count[i, j]} is the number of 
#'                  trees that both i and j are not in the tree, i.e., OOB.
#'                  Note that we replace 0 in OOB_count by 1 to avoid 0 / 0.
#'                  2) When \code{X2} is not \code{null}, assume  \code{X2} are all training data.
#'                  return (kernel / OOB_count). OOB_count[i, j] the number of trees that j-th
#'                  sample in X2 are not in the tree. So each row of OOB_count are the same.
#' @param ObsTrack  Default is \code{NULL} and use the \eqn{ObsTrack_{n_train \times ntrees}} 
#'                  for fitted RLT object. If given \code{ObsTrack}, then override it.
#' 
#' @param verbose   Whether fitting should be printed.
#' 
#' @param ... ...   Additional arguments.
#' @export

forest.kernel <- function(object,
                          X1 = NULL,
                          X2 = NULL,
                          vs.train = FALSE,
                          verbose = FALSE,
                          OOB = FALSE,
                          ObsTrack = NULL,
                          ...)
{
  if( class(object)[2] != "fit" )
    stop("object must be a fitted RLT object")

  if (is.null(X1))
    stop("self-kernel is not implemented yet.")

  if (!is.matrix(X1) & !is.data.frame(X1)) stop("X1 must be a matrix or a data.frame")
  
  if (OOB){
    vs.train = FALSE
  }
  
  # check if the ObsTrack is provided in object when train or OOB
  if (vs.train | OOB){
    if (is.null(ObsTrack)){
      # use the ObsTrack from the RLT object
      if ( is.null(object$ObsTrack) ){
        stop("Must have ObsTrack to perform vs.train or OOB. Please enable it in RLT")
      }
      else{
        ObsTrack = object$ObsTrack
      }
    }else{
      # use the given ObsTrack
      ObsTrack = ObsTrack
    }

  }


    
    # check X1 data 
    if (is.null(colnames(X1))){
      if (ncol(X1) != object$parameters$p) 
        stop("X1 dimension does not match training data, variable names are not supplied...")
    }else if (any(colnames(X1) != object$xnames)){
      warning("X1 data variables names does not match training data ...")
    
      varmatch = match(object$xnames, colnames(X1))
    
      if (any(is.na(varmatch))) 
        stop("X1 is missing some variables from the orignal training data ...")
      
      X1 = X1[, varmatch]
    }
    
    X1 <- data.matrix(X1)
    
    if (is.null(X2)){
      if (!OOB){
        K <- Kernel_Self(object$FittedForest$SplitVar,
                         object$FittedForest$SplitValue,
                         object$FittedForest$LeftNode,
                         object$FittedForest$RightNode,
                         X1,
                         object$ncat,
                         verbose)
      }else{
        # OOB self kernel for training data
        if (nrow(ObsTrack) != nrow(X1))
          stop("X1 must be the original training data for OOB trainingkernel weight")
        K <- Kernel_Self_OOB(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             X1,
                             object$ncat,
                             ObsTrack,
                             verbose);
      }
      class(K) <- c("RLT", "kernel", "self")
      
    }else{
    
      # check X2
      
      if (!is.null(X2))
        if (!is.matrix(X2) & !is.data.frame(X2))
          stop("X2 must be a matrix or a data.frame")    
      
      if (is.null(colnames(X2))){
        if (ncol(X2) != object$parameters$p) 
          stop("X2 dimension does not match training data, variable names are not supplied...")
      }else if (any(colnames(X2) != object$xnames)){
        warning("X2 data variables names does not match training data ...")
        
        varmatch = match(object$xnames, colnames(X2))
        
        if (any(is.na(varmatch))) stop("X2 missing some variables from the orignal training data ...")
        
        X2 = X2[, varmatch]
      }
      
      X2 <- data.matrix(X2)
      
      if (!vs.train){
        if (!OOB){
          # cross-kernel of X1 and X2
          K <- Kernel_Cross(object$FittedForest$SplitVar,
                            object$FittedForest$SplitValue,
                            object$FittedForest$LeftNode,
                            object$FittedForest$RightNode,
                            X1,
                            X2,
                            object$ncat,
                            verbose)
        }else{
          # cross-kernel of X1 and X2, use OOB trees for X2 
          
          if (nrow(ObsTrack) != nrow(X2))
            stop("X2 must be the original training data for OOB kernel weight")
          K <- Kernel_Cross_OOB(object$FittedForest$SplitVar,
                                object$FittedForest$SplitValue,
                                object$FittedForest$LeftNode,
                                object$FittedForest$RightNode,
                                X1,
                                X2,
                                object$ncat,
                                ObsTrack,
                                verbose)
        }
    
        class(K) <- c("RLT", "kernel", "cross")
        
      }else{
        # kernel matrix as to the training process 
        
        if (nrow(ObsTrack) != nrow(X2))
          stop("X2 must be the original training data")
        
        K <- Kernel_Train(object$FittedForest$SplitVar,
                             object$FittedForest$SplitValue,
                             object$FittedForest$LeftNode,
                             object$FittedForest$RightNode,
                             X1,
                             X2,
                             object$ncat,
                             ObsTrack,
                             verbose)
        
        class(K) <- c("RLT", "kernel", "train")
        
      }
    }
    
  
  return(K)
}
