//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Random Forest Kernel
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List Kernel_Self(arma::field<arma::ivec>& SplitVar,
                    arma::field<arma::vec>& SplitValue,
                    arma::field<arma::uvec>& LeftNode,
                    arma::field<arma::uvec>& RightNode,
                    arma::mat& X,
                    arma::uvec& Ncat,
                    size_t verbose)
{
  size_t N = X.n_rows;
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  umat K(N, N, fill::zeros);
  uvec real_id = linspace<uvec>(0, N-1, N);  
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                           SplitValue(nt),
                           LeftNode(nt),
                           RightNode(nt));
    
    // initiate all observations
    uvec proxy_id = linspace<uvec>(0, N-1, N);
    uvec TermNode(N, fill::zeros);
    
    // get terminal node id
    Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
    
    //record
    uvec UniqueNode = unique(TermNode);
    
    for (auto j : UniqueNode)
    {
      uvec ID = real_id(find(TermNode == j));
      
      K.submat(ID, ID) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
}

// [[Rcpp::export()]]
List Kernel_Cross(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& X2,
                     arma::uvec& Ncat,
                     size_t verbose)
{
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                           SplitValue(nt),
                           LeftNode(nt),
                           RightNode(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));

    for (auto j : UniqueNode)
    {
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j));
      
      K.submat(ID1, ID2) += 1;
    }
  }

  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}

// [[Rcpp::export()]]
List Kernel_Train(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& X2,
                     arma::uvec& Ncat,
                     arma::imat& ObsTrack,
                     size_t verbose)
{
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                           SplitValue(nt),
                           LeftNode(nt),
                           RightNode(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));
    ivec intreent = ObsTrack.col(nt);
    
    for (auto j : UniqueNode)
    {
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j && intreent > 0));
      
      K.submat(ID1, ID2) += 1;
    }
  }
  
  List ReturnList;
  ReturnList["Kernel"] = K;
  
  return(ReturnList);
  
}




// [[Rcpp::export()]]
List Kernel_Self_OOB(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X,
                     arma::uvec& Ncat,
                     arma::imat& ObsTrack,
                     size_t verbose){
  size_t N = X.n_rows;
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N, N, fill::zeros);
  mat OOB_count  = zeros<mat>(N, N);
  
  uvec real_id = linspace<uvec>(0, N-1, N);

  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                       SplitValue(nt),
                       LeftNode(nt),
                       RightNode(nt));
    
    // initiate all observations
    uvec proxy_id = linspace<uvec>(0, N-1, N);
    uvec TermNode(N, fill::zeros);
    
    // get terminal node ids
    Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);

    // record
    uvec UniqueNode =unique(TermNode);
    ivec intreent = ObsTrack.col(nt);
    
    // OOB_pairs: only consider OOB trees
    // transform all >= 1 values in intreent to be 0
    colvec intreent2 = conv_to<colvec>::from(intreent);
    intreent2.transform([](int val) { return val > 0 ? 1.0 : val; });
    mat OOB_pair = (1-intreent2) * (1-intreent2.t());
    OOB_count += OOB_pair;
    for (auto j : UniqueNode){
      // find the id in this node but not in this tree.
      uvec ID = real_id(find(TermNode == j && intreent == 0));
      K.submat(ID, ID) += 1;
      
    }
  }
  
  // re-normalize K by K/OOB_count. 
  // To prevent 0/0, replace 0 in OOB_count by 1
  for (uword i = 0; i < OOB_count.n_rows; i++) {
    for (uword j = 0; j < OOB_count.n_cols; j++) {
      if (OOB_count(i, j) == 0) {
        OOB_count(i, j) = 1;
      }
    }
  }
  mat K_f = conv_to<mat>::from(K);
  mat K_std = K_f / OOB_count;
  // self weight is 1
  vec d = ones<vec>(N);
  K_std.diag() = d;
  
  List ReturnList;
  ReturnList["Kernel"] = K_std;
  return(ReturnList);
  
}
// [[Rcpp::export()]]
List Kernel_Cross_OOB(arma::field<arma::ivec>& SplitVar,
                     arma::field<arma::vec>& SplitValue,
                     arma::field<arma::uvec>& LeftNode,
                     arma::field<arma::uvec>& RightNode,
                     arma::mat& X1,
                     arma::mat& X2,
                     arma::uvec& Ncat,
                     arma::imat& ObsTrack,
                     size_t verbose){
  size_t N1 = X1.n_rows;
  size_t N2 = X2.n_rows;
  size_t ntrees = SplitVar.n_elem; 
  
  // initiate output kernel
  // each element for one testing subject 
  umat K(N1, N2, fill::zeros);
  uvec real_id1 = linspace<uvec>(0, N1-1, N1);
  uvec real_id2 = linspace<uvec>(0, N2-1, N2);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    Tree_Class OneTree(SplitVar(nt),
                       SplitValue(nt),
                       LeftNode(nt),
                       RightNode(nt));
    
    // initiate all observations
    uvec proxy_id1 = linspace<uvec>(0, N1-1, N1);
    uvec proxy_id2 = linspace<uvec>(0, N2-1, N2);
    
    uvec TermNode1(N1, fill::zeros);
    uvec TermNode2(N2, fill::zeros);
    
    // get terminal node ids
    Find_Terminal_Node(0, OneTree, X1, Ncat, proxy_id1, real_id1, TermNode1);
    Find_Terminal_Node(0, OneTree, X2, Ncat, proxy_id2, real_id2, TermNode2);
    
    
    // record
    uvec UniqueNode = intersect(unique(TermNode1), unique(TermNode2));
    ivec intreent = ObsTrack.col(nt);

    
    for (auto j : UniqueNode){
      // find the id in this node but not in this tree.
      uvec ID1 = real_id1(find(TermNode1 == j));
      uvec ID2 = real_id2(find(TermNode2 == j && intreent == 0));
      K.submat(ID1, ID2) += 1;
      
    }
  }
  
  // re-normalize K by K/OOB_count. 
  // for OOB count, only count once for multiple sample
  for (uword i = 0; i < ObsTrack.n_rows; i++) {
    for (uword j = 0; j < ObsTrack.n_cols; j++) {
      if (ObsTrack(i, j) > 1) {
        ObsTrack(i, j) = 1;
      }
    }
  }
  // get the OOB count for each training sample in X2
  // replicate this count row-wise to create OOB_count_rep
  // normalize K by OOB_count_rep
  mat obs_temp = conv_to<mat>::from(ObsTrack);
  vec OOB_count = sum(1-obs_temp, 1);
  mat OOB_count_mat = repmat(OOB_count.t(), K.n_rows, 1);
  mat K_f = conv_to<mat>::from(K);
  mat K_std = K_f / OOB_count_mat;

  List ReturnList;
  ReturnList["Kernel"] = K_std;
  return(ReturnList);
  
}