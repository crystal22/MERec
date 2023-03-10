# PAKDD2023_MERec
Meta-learning Enhanced Next POI Recommendation by Leveraging Check-ins from Auxiliary Cities
-
![MERec Framework](https://github.com/oli-wang/MERec/blob/main/Figures/MERec_framework.png)

Abstract: Next point-of-interest (POI) recommendation has attracted a considerable amount of attention and become a prevalent personalized service with the development of location-based social networks. Most existing POI recommendation algorithms aim to capture user preference by employing city-level user historical check-ins, thus achieving next POI recommendation and facilitating users' exploration of the city. However, the scarcity of city-level user check-ins brings a significant challenge to user preference learning. Although prior studies attempt to mitigate this challenge by exploiting various context information, e.g., spatio-temporal and categorical information, they ignore to transfer the knowledge (i.e., common behavioral pattern) from other relevant cities (i.e., auxiliary city). In this paper, we investigate the effect of knowledge obtained from auxiliary cities and thus propose a novel Meta-learning Enhanced next POI Recommendation framework (MERec). Specifically, the MERec leverages the correlation of check-in behaviors among auxiliary cities into the meta-learning paradigm to help infer user preference in the target city. Note that it employs the user check-in patterns of the auxiliary cities by considering their cultural and geographical characteristics, so as to transfer more relevant knowledge from more correlated cities. Experiments verify the superiority of the proposed MERec against state-of-the-art algorithms. 

### Heighlights

* To the best of our knowledge, we are the first to differentiate the correlation of auxiliary cities and the target city when transferring user behavioral knowledge to enhance the next POI recommendation.
* We propose a novel meta-learning based framework, namely MERec, which leverages both the transferred knowledge and user behavioral contexts within the target city, thus alleviating the data sparsity issue.
* We conduct experiments to validate the superiority of MERec against state-of-the-art approaches.
