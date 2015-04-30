1. In fair coin flips,

       P(H) = 0.5, P(T) = 0.5
       P(H, H, H) = 0.5 * 0.5 * 0.5 = 0.125
       P(x1=x2=x3=x4) = 2 * 0.5 ^ 4 = 0.125
       P({x1,x2,x3,x4} contains 3+ heads) = 0.5 ^ 4 + 4 * 0.5 ^ 4 = 0.3125

2. Write out the formulas for joint probability, product rule of probability, chain rule of probability respectively, and Bayes’ rule.

   * Joint probability: P(A, B) = P(A) + P(B) - P(A & B) = P(A|B) * P(B)
   * Product rule of probability: P(A,B) = P(A) * P(A|B)
   * chain rule: P(A1,A2,...,An) = P(An|An-1,...,A1) * P(An-1,...,A1)
   * Bayes' rule: P(A|B) = P(B|A) * P(A) / P(B)

3. The weather is sunny or rainy. Weather of a day is only dependent to the previous. Given P(D1=sunny)=0.9, P(D2=sunny|D1=sunny)=0.8, P(D2=sunny|D1=rainy)=0.6, please specify P(D2=rainy|D1=sunny), P(D2=rainy|D1=rainy), P(D2=sunny) and P(D3=sunny).

   P(d2=r|d1=s) = 1 - P(d2=s|d1=s) = 0.2
   P(d2=r|d1=r) = 1 - P(d2=s|d1=r) = 0.4
   P(d2=s) = P(d2=s,d1=r) + P(d2=s,d1=s) = 0.9*0.8 + 0.6*0.1 = 0.78
   P(d3=s) = P(d3=s,d2=s) + P(d3=s,d2=r) = 0.78*0.8 + 0.22*0.1 = 0.646

4. What is a Bayes net?

   Bayes’ net is an efficient encoding of a probabilistic model of a domain.

5. What’s the whole joint probability of a Bayes net?

   pass...
