  //  **********************************
//  Reinforcement Learning Trees (RLT)
//  Quantile
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

//Find a split on a particular variable
void Quan_Uni_Split_Cont(Split_Class& TempSplit,
                        const uvec& obs_id,
                        const vec& x,
                        const vec& Y,
                        const vec& obs_weight,
                        double penalty,
                        size_t split_gen,
                        size_t split_rule,
                        size_t nsplit,
                        double alpha,
                        bool useobsweight,
                        Rand& rngl)
{
  size_t N = obs_id.n_elem;

  //arma::vec temp_cut_arma;
  //double temp_cut;
  //size_t temp_ind;
  double temp_score;
  
  // obs_id is already sorted by y
  vec x_temp = x(obs_id);
  vec y_temp = Y(obs_id);
  uvec sorted_id = sort_index(y_temp); 
  vec x_sub = x_temp(sorted_id);
  vec y_sub = y_temp(sorted_id);
  
  // for best split with too large node size, also use random
  bool random_split = (split_gen == 3) && (obs_id.size() > 50);

  if ((split_gen == 1) || (random_split)) // random split
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a random cut off
      size_t temp_id = obs_id( rngl.rand_sizet(0,N-1) );
      double temp_cut = x(temp_id);
        
      //temp_cut_arma = x(obs_id( rngl.rand_sizet(0,N-1) )); 
      //temp_cut = temp_cut_arma(0);

      // calculate score
      temp_score = quan_uni_cont_score_cut_sub(x_sub, y_sub, temp_cut);
      // if (useobsweight)
      //   // both use unweighted version
      //   temp_score = quan_uni_cont_score_cut_sub_w(obs_id, x, Y, temp_cut, obs_weight);
      // else
      //   temp_score = quan_uni_cont_score_cut_sub(obs_id, x, Y, temp_cut);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = temp_cut;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  // uvec indices = obs_id(sort_index(x(obs_id))); // this is the sorted obs_id
  uvec sub_id = regspace<uvec>(0, y_sub.n_elem-1);
  uvec indices = sub_id(sort_index(x_sub)); // this is the sorted obs_id
  
  // check identical 
  if ( x_sub(indices(0)) == x_sub(indices(N-1)) ) return;  
  
  // set low and high index
  size_t lowindex = 0; // less equal goes to left
  size_t highindex = N - 2;
  
  // alpha is only effective when x can be sorted
  // I need to do some changes to this
  // this will force nmin for each child node
  if (alpha > 0)
  {
    // if (N*alpha > nmin) nmin = (size_t) N*alpha;
    size_t nmin = (size_t) N*alpha;
    if (nmin < 1) nmin = 1;
    
    lowindex = nmin-1; // less equal goes to left
    highindex = N - nmin - 1;
  }
  
  // if ties
  // move index to better locations
  if ( x_sub(indices(lowindex)) == x_sub(indices(lowindex+1)) or 
    x_sub(indices(highindex)) == x_sub(indices(highindex+1)) ){
    check_cont_index_sub(lowindex, highindex, x_sub, indices);
    
    if (lowindex > highindex){
      RLTcout << "lowindex > highindex... this shouldn't happen." << std::endl;
      return;
    }
  }
  
  /*
    // if there are ties, do further check
    if ( (x(indices(lowindex)) == x(indices(lowindex + 1))) | (x(indices(highindex)) == x(indices(highindex + 1))) )
      move_cont_index(lowindex, highindex, x, indices, nmin);
    
  }else{
    // move index if ties
    while( x(indices(lowindex)) == x(indices(lowindex + 1)) ) lowindex++;
    while( x(indices(highindex)) == x(indices(highindex + 1)) ) highindex--;
    
    //If there is nowhere to split
    if (lowindex > highindex) return;
  }
  */
  
  if (split_gen == 2) // rank split
  {
    for (size_t k = 0; k < nsplit; k++)
    {
      // generate a cut off
      size_t temp_ind = rngl.rand_sizet( lowindex, highindex );
      
      // there could be ties here. need to fix this issue. 
      if ( x_sub(indices(temp_ind)) == x_sub(indices(temp_ind+1)) )
      {
        if (rngl.rand_01() > 0.5)
        { // move up
          while( x_sub(indices(temp_ind)) == x_sub(indices(temp_ind+1)) ) temp_ind++;
        }else{ // move down
          while( x_sub(indices(temp_ind)) == x_sub(indices(temp_ind+1)) ) temp_ind--;
        }
      }
      
      if (useobsweight)
        temp_score = reg_uni_cont_score_rank_sub_w(indices, y_sub, temp_ind, obs_weight);
      else
        temp_score = reg_uni_cont_score_rank_sub(indices, y_sub, temp_ind);
      
      if (temp_score > TempSplit.score)
      {
        TempSplit.value = (x(indices(temp_ind)) + x(indices(temp_ind+1)))/2 ;
        TempSplit.score = temp_score;
      }
    }
    
    return;
  }
  
  if (split_gen == 3) // best split  
  {
    // get score
    quan_uni_cont_score_best_sub(indices, x_sub, y_sub, lowindex, highindex,
                                 TempSplit.value, TempSplit.score);
    /*
    if (useobsweight){
      reg_uni_cont_score_best_sub_w(indices, x, Y, lowindex, highindex,
                                    TempSplit.value, TempSplit.score, obs_weight);
    } 
    else{
      reg_uni_cont_score_best_sub(indices, x_sub, y_sub, lowindex, highindex,
                                  TempSplit.value, TempSplit.score);
    }
    */
    return;
  }
  
}

// calculate re-normalized Kolmogorov-Smirnov:
double get_ks_score(vec& samples1, vec& samples2){
  // calculate re-normalized Kolmogorov-Smirnov:
  // max{F1(x), F2(x)} / (sqrt(1/N1 + 1/N2)), where F is the emprical CDF
  // calculate the size, sort residuals
  // sample1 and sample2 are sorted
  size_t NL = samples1.size();
  size_t NR = samples2.size();
  
  double max_diff = 0;
  double curr_diff;
  size_t l = 0;
  size_t r = 0;
  while (l < NL && r < NR){
    if ( samples1(l) < samples2(r) ){
      l++;
    }else{
      r++;
    }
    // calculate the difference of two CDF at current cutoff
    curr_diff = abs(l / (double)NL - r / (double)NR);
    if (curr_diff > max_diff){
      max_diff = curr_diff;
    }
  }
  double normalized_ks = max_diff / (sqrt(1/(double)NL + 1/(double)NR));
  return normalized_ks;
}


//Calculate a KS score at a random cut
double quan_uni_cont_score_cut_sub(const vec& x_sub,
                                   const vec& y_sub,
                                   double a_random_cut){
  // left_branch: all y(obs_id) whose x <= a_random_cut; else: right_branch
  // x_sub: a len(n) vector of 1D feature (sorted by y's order)
  // y_sub: a len(n) vector of residuals or outcomes (sorted)
  // obs_id: a vector of selected observations 
  
  uvec left_id = find(x_sub <= a_random_cut);
  uvec right_id = find(x_sub > a_random_cut);
  
  // check if there is no observation in the left or right
  size_t left_count = left_id.size();
  size_t right_count = right_id.size();
  if (left_count == 0 || right_count == 0){
    return -1;
  }
  
  // calculate re-normalized Kolmogorov-Smirnov score:
  // max{F1(x) - F2(x)} / (sqrt(1/N1 + 1/N2))
  vec y_sub_l = y_sub(left_id);
  vec y_sub_r = y_sub(right_id);
  double score = get_ks_score(y_sub_l, y_sub_r);
  return score;
}


//For best split
void quan_uni_cont_score_best_sub(uvec& indices,
                                  const vec& x,
                                  const vec& Y,
                                  size_t lowindex, 
                                  size_t highindex, 
                                  double& temp_cut, 
                                  double& temp_score){
  
  double score = 0;
  size_t N = indices.size();
  
  //Trying the other splits
  for (size_t i = lowindex; i <= highindex; i++){
    uvec left_id = find(x <= x(indices(i)));
    uvec right_id = find(x > x(indices(i)));
    vec y_sub_l = Y(left_id);
    vec y_sub_r = Y(right_id);
    score = get_ks_score(y_sub_l, y_sub_r);
    
    //If the score has improved, find cut and set new score
    if (score > temp_score){
      temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
      temp_score = score;
    }
  }
  return;
}