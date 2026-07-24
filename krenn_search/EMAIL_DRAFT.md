# Draft email to Mario Krenn (not sent — for review)

Subject: A d=4 k-monochromatic graph (Question 2), with exact + Lean-verified certificate

Dear Prof. Krenn,

Regarding the inherited-vertex-coloring questions on your prize page: we found
a k-monochromatic graph (Question 2 of Krenn-Gu-Soltész) with d = 4 colors —
to our knowledge the first with d > 3 in any parameter regime. In quantum
terms it is a 4-dimensional 4-photon GHZ state heralded by 2 ancilla photons.

n = 6 (vertices 1-4 state, 5-6 heralds), k = 4, all edges monochromatic,
all weights rational:

  red:  w(1,2)=w(1,3)=-1/2, w(2,4)=w(3,4)=1/2, w(1,5)=1, w(4,5)=-1,
        w(5,6)=2, w(1,6)=w(2,6)=w(3,6)=w(4,6)=1
  c1:   w(1,3)=1/2, w(2,4)=1
  c2:   w(1,2)=1/2, w(3,4)=1
  c3:   w(1,4)=1/2, w(2,3)=1

The four colorings (a,a,a,a,red,red) each get weight exactly 1, and all other
4092 inherited vertex colorings cancel exactly. Certificates:
 1. an independent symbolic verifier (sympy, exact rational arithmetic over
    all 15 perfect matchings x 4^6 colorings);
 2. a Lean 4 formalization whose main theorem is checked by the Lean kernel
    alone (`decide`, no native-code trust).

Both are attached / available at [repository link].

Two questions: (1) is such a d=4 instance known — we could not find one in
the literature (the 2019 paper lists d=3 as the maximum among known
k-monochromatic graphs); (2) if it is new, does it fall within the scope of
the January 2021 reward update concerning Question 2 solutions, and would a
short journal note be the appropriate vehicle?

Best regards,
[name]
