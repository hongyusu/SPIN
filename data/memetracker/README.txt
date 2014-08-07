Sample Data Files
  	The following data files were extracted from the MemeTracker dataset. Please read the instructions below on how to use them.

  	memeS.sn (2.7MB)    memeS.out (285kb)
  	memeM.sn (3.8MB)   memeM.out (534kb)
  	memeL.sn (6.7MB)    memeL.out (1.3MB)
 
  	Files with extension ".sn" define a social network structure. Each line corresponds to one directed arc between two nodes. For example, line
NodeA    NodeB
signifies that there is a directed arc from 'NodeA' to 'NodeB'. This, in turn, means that 'NodeA' influences 'NodeB', with 'influence' defined in terms of the Independent Cascade Model (ICM - see paper). 	 
 
  	Files with extension ".out" contain a set of propagations. Each propagation, in turn, is defined as a sequence of activations. A propagation begins with the activation of node 'omega', a special node that models sources of influence outside the network. In the data files, activations of 'omega' are represented with the following line,
             omega   0
which signifies that node 'omega' is activated at time 0. Every such line signifies the beginning of another propagation. Lines between 'omega' activations correspond to a single propagation, and each of them corresponds to the activation of one node. For example, line
NodeA    NodeB   T
signifies that 'NodeA' successfully influenced 'NodeB' to perform an action at time T. In our paper, we assume that 'NodeA' is generally not known. Therefore, for the current implementation of Spine, 'NodeA' acts simply as a placeholder. This format was chosen in interest of consistency with possible extensions of 'Spine' that assume that this information is available. In the provided data files, 'NodeA' is uniformly set to 'omega'.

"mainstreamDescr" contains the memes in our dataset (meme id + meme text -- one meme per line)
"mainstreamLog" contains the blogs or websites that posted text with a meme from our dataset (meme id + timestamp in milliseconds + blog/website that posted meme -- one post per line).
All memes were generated in March 2009.

We were not given an explicit social network as input, but we inferred a social network from propagations (we assumed that a blog/website A that posted a meme first influenced a blog/website B that posted the same meme later - and thus we inferred a non-zero probability arc (A,B)).

