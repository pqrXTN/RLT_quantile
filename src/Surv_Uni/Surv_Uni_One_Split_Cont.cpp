//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Split_Cont(Uni_Split_Class& TempSplit, 
                         uvec& obs_id,
                         const vec& x,
                         const uvec& Y, // Y is collapsed
                         const uvec& Censor, // Censor is collapsed
                         double penalty,
                         int split_gen,
                         int split_rule,
                         int nsplit,
                         size_t nmin, 
                         double alpha,
                         vec& obs_weight,
                         bool useobsweight,
                         size_t NFail,
                         int failforce)
{
    size_t N = obs_id.n_elem;
    
    arma::vec temp_cut_arma;
    double temp_cut;
    size_t temp_ind;
    double temp_score;
    
    if (split_gen == 1) // random split
    {
        DEBUG_Rcout << "      --- Surv_One_Split_Cont with " << nsplit << " random split " << std::endl;

        for (int k = 0; k < nsplit; k++)
        {
            // generate a random cut off
            temp_cut_arma = x(obs_id( (size_t) intRand(0, N-1) ));
            temp_cut = temp_cut_arma(0);
            
            if (useobsweight){
                temp_score = surv_cont_score_at_cut_w(obs_id, x, Y, Censor, NFail, temp_cut, obs_weight, split_rule);
            }else{
                temp_score = surv_cont_score_at_cut(obs_id, x, Y, Censor, NFail, temp_cut, split_rule);
            }
            
            if (temp_score > TempSplit.score)
            {
                TempSplit.value = temp_cut;
                TempSplit.score = temp_score;
            }
        }
        
        DEBUG_Rcout << "      --- Best cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
        return;
    }
    
    
    // alpha is only effective when x can be sorted
    if (N*alpha > nmin) nmin = (size_t) N*alpha;
    
    uvec obs_ranked = sort_index(x(obs_id)); // this is the sorted obs_id
    uvec indices = obs_id(obs_ranked); // this is the sorted obs_id
    
    // check identical 
    if ( x(indices(0)) == x(indices(N-1)) ) return;
    
    // set low and high index
    size_t lowindex = nmin - 1; // less equal goes to left
    size_t highindex = N - nmin - 1;
    
    // if there are ties, do further check
    if ( (x(indices(lowindex)) == x(indices(lowindex + 1))) | (x(indices(highindex)) == x(indices(highindex + 1))) )
        move_cont_index(lowindex, highindex, x, indices, nmin);
    
    DEBUG_Rcout << "      --- lowindex " << lowindex << " highindex " << highindex << " sample size " << N << std::endl;
    
    if (split_gen == 2) // rank split
    {
        DEBUG_Rcout << "      --- Surv_One_Split_Cont with " << nsplit << " rank split " << std::endl;
        
        for (int k = 0; k < nsplit; k++)
        {
            // generate a cut off
            temp_ind = intRand(lowindex, highindex);
            
            if (useobsweight)
                temp_score = surv_cont_score_at_index_w(obs_ranked, Y, Censor, NFail, temp_ind, obs_weight, split_rule);
            else
                temp_score = surv_cont_score_at_index(obs_ranked, Y, Censor, NFail, temp_ind, split_rule);
            
            if (temp_score > TempSplit.score)
            {
                TempSplit.value = (x(indices(temp_ind)) + x(indices(temp_ind+1)))/2 ;
                TempSplit.score = temp_score;
            }
        }
        
        DEBUG_Rcout << "      --- Rank cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
        return;
    }
    
    
    if (split_gen == 3) // best split  
    {
        DEBUG_Rcout << "      --- Reg_One_Split_Cont with best split, total sample " << x.size() << std::endl;
        
        if (useobsweight)
            surv_cont_score_best_w(indices, x, obs_ranked, Y, Censor, NFail, lowindex, highindex, TempSplit.value, TempSplit.score, obs_weight, split_rule);
        else
            surv_cont_score_best(indices, x, obs_ranked, Y, Censor, NFail, lowindex, highindex, TempSplit.value, TempSplit.score, split_rule);
        
        
        DEBUG_Rcout << "      --- Best cut off at " << TempSplit.value << " with score " << TempSplit.score << std::endl;
        
        return;
        
    }
}


double surv_cont_score_at_cut(uvec& obs_id,
                              const vec& x,
                              const uvec& Y,
                              const uvec& Censor,
                              size_t NFail,
                              double a_random_cut,
                              int split_rule)
{
    // x is of full length, Y is collapsed, length same as id
    
    uvec Left_Count_Fail(NFail+1, fill::zeros);
    uvec Left_Count_Censor(NFail+1, fill::zeros);
    uvec Right_Count_Fail(NFail+1, fill::zeros);
    uvec Right_Count_Censor(NFail+1, fill::zeros);
    
    size_t LeftN = 0;     
    size_t N = obs_id.n_elem;
    
    // initiate the failure and censoring counts
    for (size_t i = 0; i<obs_id.n_elem; i++)
    {
        if (x[obs_id[i]] <= a_random_cut) // go left
        {
            if (Censor[i] == 1)
                Left_Count_Fail[Y[i]]++;
            else
                Left_Count_Censor[Y[i]]++;
            
            LeftN++;
        }else{  // go right
            if (Censor[i] == 1)
                Right_Count_Fail[Y[i]]++;
            else
                Right_Count_Censor[Y[i]]++;
        }
    }
    
    if (split_rule == 1)
        return logrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, (double) LeftN, (double) N, NFail);
    
    if (split_rule == 2)
        return suplogrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, (double) LeftN, (double) N, NFail);
    
    Rcout << "      --- splitting rule not implemented yet " << std::endl;
}


double surv_cont_score_at_cut_w(uvec& obs_id,
                                const vec& x,
                                const uvec& Y,
                                const uvec& Censor,
                                size_t NFail,
                                double a_random_cut,
                                vec& obs_weight,
                                int split_rule)
{
    Rcout << "      --- weighted surv split at random cut not implemented yet " << std::endl;
    
    return -1;
}


double surv_cont_score_at_index(uvec& obs_ranked,
                                const uvec& Y, 
                                const uvec& Censor, 
                                size_t NFail,
                                size_t a_random_ind,
                                int split_rule)
{
    // Y is collapsed, length same as obs_ranked
    
    uvec Left_Count_Fail(NFail+1, fill::zeros);
    uvec Left_Count_Censor(NFail+1, fill::zeros);
    uvec Right_Count_Fail(NFail+1, fill::zeros);
    uvec Right_Count_Censor(NFail+1, fill::zeros);
    
    size_t LeftN = 0;     
    size_t N = obs_ranked.n_elem;
    
    // initiate the failure and censoring counts
    for (size_t i = 0; i<= a_random_ind; i++)
    {
        if (Censor(obs_ranked(i)) == 1)
            Left_Count_Fail(Y(obs_ranked(i)))++;
        else
            Left_Count_Censor(Y(obs_ranked(i)))++;
        
        LeftN++;
    }
    
    for (size_t i = a_random_ind+1; i < N; i++) 
    {
        if (Censor(obs_ranked(i)) == 1)
            Right_Count_Fail(Y(obs_ranked(i)))++;
        else
            Right_Count_Censor(Y(obs_ranked(i)))++;
    }
    
    if (split_rule == 1)
        return logrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, (double) LeftN, (double) N, NFail);
    
    if (split_rule == 2)
        return suplogrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, (double) LeftN, (double) N, NFail);
    
    Rcout << "      --- splitting rule not implemented yet " << std::endl;

}

double surv_cont_score_at_index_w(uvec& obs_ranked,
                                  const uvec& Y, 
                                  const uvec& Censor, 
                                  size_t NFail,
                                  size_t a_random_ind,
                                  vec& obs_weight,
                                  int split_rule)
{
    Rcout << "      --- weighted surv split at index cut not implemented yet " << std::endl;
    
    return -1;

}


double surv_cont_score_best(uvec& indices, // for x, sorted
                            const vec& x,
                            uvec& obs_ranked, // for Y and censor, sorted
                            const uvec& Y, 
                            const uvec& Censor, 
                            size_t NFail, 
                            size_t lowindex, 
                            size_t highindex, 
                            double& temp_cut, 
                            double& temp_score,
                            int split_rule)
{
    double score = 0;
    
    // Y is collapsed, length same as indices
    
    uvec Left_Count_Fail(NFail+1, fill::zeros);
    uvec Left_Count_Censor(NFail+1, fill::zeros);
    uvec Right_Count_Fail(NFail+1, fill::zeros);
    uvec Right_Count_Censor(NFail+1, fill::zeros);
    
    size_t N = indices.n_elem;
    
    // initiate the failure and censoring counts
    for (size_t i = 0; i<= lowindex; i++)
    {
        if (Censor(obs_ranked(i)) == 1)
            Left_Count_Fail(Y(obs_ranked(i)))++;
        else
            Left_Count_Censor(Y(obs_ranked(i)))++;
    }
    
    for (size_t i = lowindex+1; i < N; i++)
    {
        if (Censor(obs_ranked(i)) == 1)
            Right_Count_Fail(Y(obs_ranked(i)))++;
        else
            Right_Count_Censor(Y(obs_ranked(i)))++;
    }
    
    for (size_t i = lowindex; i <= highindex; i++)
    {
        
        while (x(indices(i)) == x(indices(i+1))){
            i++;
            
            if (Censor(obs_ranked(i)) == 1)
                Left_Count_Fail(Y(obs_ranked(i)))++;
            else
                Left_Count_Censor(Y(obs_ranked(i)))++;
            
            
            if (Censor(obs_ranked(i)) == 1)
                Right_Count_Fail(Y(obs_ranked(i)))--;
            else
                Right_Count_Censor(Y(obs_ranked(i)))--;

        }
        
        if (split_rule == 1)
            score = logrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, (double) i + 1, (double) N, NFail);
        
        if (split_rule == 2)
            score = suplogrank(Left_Count_Fail, Left_Count_Censor, Right_Count_Fail, Right_Count_Censor, (double) i + 1, (double) N, NFail);

        if (score > temp_score)
        {
            temp_cut = (x(indices(i)) + x(indices(i + 1)))/2 ;
            temp_score = score;
        }
        
        if (i + 1 <= highindex)
        {
            if (Censor(obs_ranked(i+1)) == 1)
                Left_Count_Fail(Y(obs_ranked(i+1)))++;
            else
                Left_Count_Censor(Y(obs_ranked(i+1)))++;
            
            
            if (Censor(obs_ranked(i+1)) == 1)
                Right_Count_Fail(Y(obs_ranked(i+1)))--;
            else
                Right_Count_Censor(Y(obs_ranked(i+1)))--;
        }
    }
}


double surv_cont_score_best_w(uvec& indices,
                              const vec& x,
                              uvec& obs_ranked,
                              const uvec& Y, 
                              const uvec& Censor, 
                              size_t NFail, 
                              size_t lowindex, 
                              size_t highindex, 
                              double& temp_cut, 
                              double& temp_score,
                              vec& obs_weight,
                              int split_rule)
{
    Rcout << "      --- weighted surv split at best not implemented yet " << std::endl;
    
    return -1;
}







double logrank(uvec& Left_Count_Fail,
               uvec& Left_Count_Censor,
               uvec& Right_Count_Fail,
               uvec& Right_Count_Censor,
               double LeftN,
               double N,
               size_t NFail)
{
    double numerator = 0;
    double denominator = 0;
    double tempscore = -1;
    
    // calculate the logrank for this split
    LeftN -= Left_Count_Censor[0];
    N -= Left_Count_Censor[0] + Right_Count_Censor[0];    
    
    for (size_t j = 1; j <= NFail && N > 1; j++)
    {
        numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/N - Left_Count_Fail[j];
        denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/N*(1- LeftN/N)*(N - Left_Count_Fail[j] - Right_Count_Fail[j])/(N - 1);

        LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
        N -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
    }
    
    if (denominator > 0)
        tempscore = numerator*numerator/denominator;
    
    return tempscore;
}

double suplogrank(uvec& Left_Count_Fail,
                  uvec& Left_Count_Censor,
                  uvec& Right_Count_Fail,
                  uvec& Right_Count_Censor,
                  double LeftN,
                  double N,
                  size_t NFail)
{
    Rcout << "      --- suplogrank not implemented yet " << std::endl;

}


