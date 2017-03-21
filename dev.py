import sys

EXAMPLE_APPLICATION_BINARY = "/Users/dev/code/hsvm/hsvm"
BINARY_NINJA_API_PATH = "/Applications/Binary Ninja.app/Contents/Resources/python"
BINARY_NINJA_PLUGINS_PATH = "/Users/dev/Library/Application Support/Binary Ninja/plugins"

sys.path.append(BINARY_NINJA_API_PATH)
sys.path.append(BINARY_NINJA_PLUGINS_PATH)

from binaryninja import *
import varslice

bv = BinaryViewType.get_view_of_file(EXAMPLE_APPLICATION_BINARY)

bb = bv.get_basic_blocks_at(0x1148)[0]

from varslice.reachingdefinitions import ReachingDefinitions
llil_graph = varslice.graph_function_llil_instructions(bb.function)
reachingDefinitions = ReachingDefinitions(llil_graph, bv).definitions

print reachingDefinitions
