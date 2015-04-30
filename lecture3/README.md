1. Explain Minimax strategy.

   Minimax strategy is that assuming utility of the opponent's strategy and choosing the strategy that has the minimal risk.

2. Explain α-β pruning strategy.

   α-β pruning strategy is that when using Minimax and encountering the utility of the branch is smaller than the present utility, the other leaf of the branch could be safely ignored.

3. Explain Dominant Strategy, Pretto optimal and Nash Equilibrium in the following Prisoner's dilemma if they exists.

                      Alice:testify      Alice:refuse
       Bob:testify       -5,-5             -10,0
       Bob:refuse         0,-10             -1,-1

   * Dominant Strategy: both testify (does better no matter what other player does)
   * Pareto Optimal: both refuse (no other outcome that all players would prefer)
   * Nash Equilibirium: both testify (no player can benefit from switching)

4. Calculate the best utility of the best strategy. See problem 4 of `minimax2.py`.

5. Calculate the best utility of the best strategy. See problem 5 of `minimax2.py`.