/-
Exact verification of a d = 4 k-monochromatic graph (n = 6, k = 4) —
an affirmative instance of Question 2 of Krenn, Gu, Soltész,
"Questions on the Structure of Perfect Matchings inspired by Quantum Physics"
(https://mariokrenn.wordpress.com/graph-theory-question/).

Vertices 0-3 carry the state, vertices 4, 5 are heralds. Colors: 0 = red
(herald color), 1, 2, 3. All edges are monochromatic; the rational solution
is scaled by 2 so all weights are integers, which multiplies every perfect
matching product (3 edges) by 8. The k-monochromatic conditions become:
  * the four colorings (a,a,a,a,0,0), a < 4, have class weight 8,
  * every other of the 4^6 inherited vertex colorings has class weight 0.
The theorem below checks all 4096 colorings against all 15 perfect matchings
of K6 by computation.

Checked with Lean 4.32.1. The proof is by `decide` — a kernel-only
computation (~3.5 min); replace with `native_decide` for a ~2 s check that
additionally trusts the Lean compiler.
-/

set_option maxHeartbeats 400000000
set_option maxRecDepth 100000

namespace KrennQ2D4

/-- Integer edge weights (2 x the rational solution). `w i j c` is the weight
of edge {i,j} (i < j) carrying color `c` at both endpoints; absent edges 0. -/
def w (i j c : Nat) : Int :=
  if i = 0 ∧ j = 1 ∧ c = 0 then -1 else
  if i = 0 ∧ j = 1 ∧ c = 2 then 1 else
  if i = 0 ∧ j = 2 ∧ c = 0 then -1 else
  if i = 0 ∧ j = 2 ∧ c = 1 then 1 else
  if i = 0 ∧ j = 3 ∧ c = 3 then 1 else
  if i = 0 ∧ j = 4 ∧ c = 0 then 2 else
  if i = 0 ∧ j = 5 ∧ c = 0 then 2 else
  if i = 1 ∧ j = 2 ∧ c = 3 then 2 else
  if i = 1 ∧ j = 3 ∧ c = 0 then 1 else
  if i = 1 ∧ j = 3 ∧ c = 1 then 2 else
  if i = 1 ∧ j = 5 ∧ c = 0 then 2 else
  if i = 2 ∧ j = 3 ∧ c = 0 then 1 else
  if i = 2 ∧ j = 3 ∧ c = 2 then 2 else
  if i = 2 ∧ j = 5 ∧ c = 0 then 2 else
  if i = 3 ∧ j = 4 ∧ c = 0 then -2 else
  if i = 3 ∧ j = 5 ∧ c = 0 then 2 else
  if i = 4 ∧ j = 5 ∧ c = 0 then 4 else
  0

/-- The 15 perfect matchings of K6. -/
def pms : List (List (Nat × Nat)) := [
  [(0, 1), (2, 3), (4, 5)],
  [(0, 1), (2, 4), (3, 5)],
  [(0, 1), (2, 5), (3, 4)],
  [(0, 2), (1, 3), (4, 5)],
  [(0, 2), (1, 4), (3, 5)],
  [(0, 2), (1, 5), (3, 4)],
  [(0, 3), (1, 2), (4, 5)],
  [(0, 3), (1, 4), (2, 5)],
  [(0, 3), (1, 5), (2, 4)],
  [(0, 4), (1, 2), (3, 5)],
  [(0, 4), (1, 3), (2, 5)],
  [(0, 4), (1, 5), (2, 3)],
  [(0, 5), (1, 2), (3, 4)],
  [(0, 5), (1, 3), (2, 4)],
  [(0, 5), (1, 4), (2, 3)]
]

/-- Color of vertex `v` in the coloring encoded by `code` (base-4 digits). -/
def chi (code v : Nat) : Nat := code / 4 ^ (5 - v) % 4

/-- Weight contributed by edge `e` under coloring `code` (0 unless the edge is
monochromatic under the coloring). -/
def edgeW (code : Nat) (e : Nat × Nat) : Int :=
  if chi code e.1 = chi code e.2 then w e.1 e.2 (chi code e.1) else 0

/-- Weight of one perfect matching under a coloring. -/
def pmW (code : Nat) (M : List (Nat × Nat)) : Int :=
  M.foldl (fun a e => a * edgeW code e) 1

/-- Class weight of a coloring: sum over all perfect matchings. -/
def clsW (code : Nat) : Int :=
  pms.foldl (fun a M => a + pmW code M) 0

/-- Codes of the four k-monochromatic colorings (a,a,a,a,0,0):
`a * (4^5 + 4^4 + 4^3 + 4^2) = a * 1360`. -/
def target (code : Nat) : Int :=
  if code = 0 ∨ code = 1360 ∨ code = 2720 ∨ code = 4080 then 8 else 0

/-- **Main theorem**: the graph is k-monochromatic with n = 6, k = 4, d = 4. -/
theorem kmono_d4_verified :
    (List.range 4096).all (fun code => clsW code == target code) = true := by
  decide

end KrennQ2D4
