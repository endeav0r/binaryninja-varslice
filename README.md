# Variable Slice (v0.1 alpha)
Author: **Alex Eubanks**

## Description:

This plugin wraps Binary Ninja with a graph, and allows for performing analysis over the graph.

Currently implemented are:
  * Highlight Dominators - Highlight all of a block's dominators.
  * Highlight Immediate Dominators - Highlight a block's immediate dominator.
  * Highlight Innermost Loop - Highlight the innermost loop of a block if that block is part of a loop.
  * Highlight Loop Branch - Attempts to highlight the branching instruction for a loop.
  * Highlight Predecessors - Highlight all predecessor blocks of a block.
  * Loop Detection - Provides a report of the number of loops detected by function.

I eventually want to slice a variable's use through a function, but I think I'm going to wait for binaryninja crew to implement SSA first. There is, however, a healthy amount of code for a preliminary SSA-implementation in this code base.

## Minimum Version

This plugin requires the following minimum version of Binary Ninja:

 * dev - 1.0.dev-679

## Required Dependencies

None. Why would you want dependencies?

## License

This plugin is released under a [MIT](LICENSE) license.
