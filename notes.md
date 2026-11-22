# Resources

## setup

1. On laptop, download `setup_mac.sh` and `arena_fall_shared` from Slack into
   `~/Downloads/`. Run the setup script (only required first time)
2. Run `ssh gpu-cedar`
3. Clone this repository `https://github.com/lairdstewart/ARENA_materials`
4. `./install_uv.sh`
5. In VS code, do command+shift+p, remote-ssh connect to host, then pick cpu-cedar
6. command+shift+p, Python: select interpreter, ./venv/bin/python
7. select kernel -> python environments -> pick that kernel
8. To push changes, copy personal access token when it asks for password

Resources

- #link(
  "https://github.com/akhilvasvani/Linear-Algebra-Basics/tree/master",
  )[Linear Algebra Basics for ML/DL]
- #link("https://explained.ai/matrix-calculus/")[Matrix Calculus for Deep
  Learning]

# November 7th

arena math prerecs: linear algebra.

Things I learned

- Column visualisation of system of linear equations.
- Story about data science w/ text is not good
- Every conversation is an interview
- One explanation for why the determinant uses eigenvalues is that the SVD is a
  rotation, scale, rotation and the scale part is a diagona matrix that does the
  scaling. And The determinant represents how much the space is scaled/shrunk.

Einstein notation rules:

1. A letter appearing twice in a term is summed over all its values.
2. A letter appearing once is kept and becomes a dimension of the result.
3. The same letter takes the same value everywhere it appears in the term.
4. There is no row/column distinction; matching letters alone determine what
   gets multiplied.
5. In an equation, both sides must have the same kept letters.

Today I (1) finished chapter 2 of https://www.deeplearningbook.org/ to refresh
my linear algebra (2) reviewed KL divergence (3) reviewed einops/Einstein
notation.

Next time i plan to finish part0_prereqs and start on part_1 raytracing

# November 14th

## Einops

To grok einops composition, start with a simpler example with letters. First,
imagine going from 2D to 1D where the og matrix has width = 2 and height = 3

```
(h x w) = (3 x 2)
a b
c d
e f

h w -> (w h)
a c e b d f

h w -> (h w)
a b c d e f
```

And note that the right-most dimension in the parenthesis changes the fastest.
This is like normal numbers or like truth tables from CompE classes. Slightly
more complex (note I'm writing 2D matrices with rows as dimension 1 and cols as
dimension 2)

```
(b x h x w) = (3 x 3 x 2)
a b    g h    m n
c d    i j    o p
e f    k l    q r

b h w -> b (w h)
a c e b d f
g i k h j l
m o q n p r

b h w -> b (h w)
a b c d e f
g h i j k l
m n o p q r
```

It's like changing the order of a for loop when you are storing a matrix
linearly in memory in a computer engineering class.

the distinction between repeat, reduce, and rearrange is the number of elements
in the resulting tensor. It is more, less, and the same respectively.

## broadcasting

1. You can prepend dummy dimensions (of size 1) to the start of a tensor until
   both have the same number of dimensions
2. After this point, if some dimension has size 1 in one of the tensors, it can
   be repeated until it matches the size of the corresponding dimension in the
   other tensor

# November 21st

Goal today is to get the GPUs setup then start on

what i did

- Setup ssh/gpu
- Arena lecture week 1 day 1: https://www.youtube.com/embed/11Z50mi8dSg
- Neel nanda's video on "a mathematical framework for transformer interp" https://www.youtube.com/watch?v=KV5gbOmHbjU
  very good.

Things i learned

- tokenization of long math equations is sort of a mess, no wonder early small
  models struggled with math.
- neel nanda's explanation
  - Neel nanda's explanation of position is simply that the residual stream is
    `d` dimensional, and the model is obviously representing much more than `d`
    ideas, so it can't have every "idea" be orthogonal in this `d` dimensional
    space. there is the nice result about how if you don't require total
    orthogonality, but like dot product of 0.1 between vectors then you can
    squeeze in exponentially more semi-orthogonal directions into this space.
    Hadn't heard superposition described in that way before.
  - From need nanda's video, there is an idea that some deeper layers do
    "residual stream cleaning" where if previous layers used some direction within
    the residual stream to communicate with one aother but that communication is
    no longer needed, it can subtract out that direction/message to "clean up" the
    residual stream. woah. https://www.youtube.com/watch?v=KV5gbOmHbjU
  - Thinking of main transformer head equation as `AXW_v^TW_o^T` is useful. It
    shows that W_v and W_o need to be thought of together as a single matrix since
    they are multiplied by eachother.
