import sys

EXAMPLE_APPLICATION_BINARY = "/Users/endeavor/code/complexity/example0"
BINARY_NINJA_API_PATH = "/Applications/Binary Ninja.app/Contents/Resources/python"
BINARY_NINJA_PLUGINS_PATH = "/Users/endeavor/Library/Application Support/Binary Ninja/plugins"

sys.path.append(BINARY_NINJA_API_PATH)
sys.path.append(BINARY_NINJA_PLUGINS_PATH)

from binaryninja import *
import varslice

bv = BinaryViewType.get_view_of_file(EXAMPLE_APPLICATION_BINARY)

bb = bv.get_basic_blocks_at(0x100000f2e)[0]

graph = varslice.graph_function(bb.function)

dominators = graph.compute_dominators()

print 'dominators'
for index in dominators :
    print hex(index), map(lambda x: hex(x), dominators[index])

print 'immediate dominators'
immediate_dominators = graph.compute_immediate_dominators()
for index in immediate_dominators :
    print hex(index), hex(immediate_dominators[index])

print 'loops'
loops = graph.detect_loops()
for loop in loops :
    print map(lambda x: hex(x), loop)

print 'highlight loop branch'
varslice.highlight_loop_branch(bv, 0x100000f2e)

print 'detect loops'
for function in bv.functions :
    graph = varslice.graph_function(function)
    print function.name, len(graph.detect_loops())
