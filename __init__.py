from binaryninja import *
import copy

from graph import Graph, find_loop_dominator
from reachingdefinitions import ReachingDefinitions

def graph_function_llil_instructions (function) :
    '''
    Returns a graph where each LLIL instruction is a vertex in the graph
    '''
    graph = Graph(0)

    # Get the low_level_il basic blocks
    basic_blocks = function.low_level_il.basic_blocks

    # Add all the low_level_il instructions as their own vertices
    for basic_block in basic_blocks :
        for ins in basic_block :
            graph.add_vertex(ins.instr_index, ins)

    # Go back through and add edges
    for basic_block in basic_blocks :
        # Add edges between instructions in block
        previous_ins = None
        for ins in basic_block :
            if previous_ins == None :
                previous_ins = ins.instr_index
                continue
            graph.add_edge_by_indices(previous_ins, ins.instr_index)
            previous_ins = ins.instr_index
        # Add edges between basic blocks
        for outgoing_edge in basic_block.outgoing_edges :
            target = outgoing_edge.target.start
            graph.add_edge_by_indices(previous_ins, target)

    return graph


def graph_function_low_level_il (function) :
    '''
    Returns a graph where each LLIL Basic Block is a vertex in the graph
    '''
    graph = Graph(0)

    # get the low_level_il basic blocks
    basic_blocks = function.low_level_il.basic_blocks

    # We are going to add each basic block to our graph
    for basic_block in basic_blocks :
        graph.add_vertex(basic_block.start, basic_block)

    # Now we are going to add all the edges
    for basic_block in basic_blocks :
        for outgoing_edge in basic_block.outgoing_edges :
            target = outgoing_edge.target
            graph.add_edge_by_indices(basic_block.start, target.start, None)

    # Now return the graph
    return graph


def graph_function (function) :
    '''
    Returns a graph where each basic block is a vertex in the graph
    '''
    graph = Graph(function.start)

    basic_blocks = function.basic_blocks

    for basic_block in basic_blocks :
        graph.add_vertex(basic_block.start, basic_block)

    for basic_block in basic_blocks :
        for outgoing_edge in basic_block.outgoing_edges :
            target = outgoing_edge.target
            graph.add_edge_by_indices(basic_block.start, target.start, None)

    return graph


def highlight_predecessors (bv, address) :
    bb = bv.get_basic_blocks_at(address)[0]
    graph = graph_function(bb.function)

    # Let's start by clearing al the basic block highlights.
    for bb_ in graph.get_vertices_data() :
        bb_.set_user_highlight(HighlightStandardColor.NoHighlightColor)

    # Get this block's predecessors.
    predecessors = graph.compute_predecessors()[bb.start]

    # Highlight all predecessors blue.
    for predecessor in predecessors :
        bb = graph.get_vertex_from_index(predecessor).data
        bb.set_user_highlight(HighlightStandardColor.BlueHighlightColor)

    # We'll go ahead and highlight the target veretx as well
    bv.get_basic_blocks_at(address)[0].set_user_highlight(HighlightStandardColor.BlueHighlightColor)


def highlight_dominators (bv, address) :
    bb = bv.get_basic_blocks_at(address)[0]
    graph = graph_function(bb.function)

    # Let's start by clearing al the basic block highlights.
    for bb_ in graph.get_vertices_data() :
        bb_.set_user_highlight(HighlightStandardColor.NoHighlightColor)

    # Get this block's predecessors.
    dominators = graph.compute_dominators()[bb.start]

    # Highlight all predecessors blue.
    for dominator in dominators :
        bb = graph.get_vertex_from_index(dominator).data
        bb.set_user_highlight(HighlightStandardColor.GreenHighlightColor)


def highlight_immediate_dominator (bv, address) :
    bb = bv.get_basic_blocks_at(address)[0]
    graph = graph_function(bb.function)

    # Let's start by clearing al the basic block highlights.
    for bb_ in graph.get_vertices_data() :
        bb_.set_user_highlight(HighlightStandardColor.NoHighlightColor)

    # Get this block's predecessors.
    immediate_dominator = graph.compute_immediate_dominators()[bb.start]

    bb = graph.get_vertex_from_index(immediate_dominator).data
    bb.set_user_highlight(HighlightStandardColor.CyanHighlightColor)


def highlight_innermost_loop (bv, address) :
    bb = bv.get_basic_blocks_at(address)[0]
    graph = graph_function(bb.function)

    # Let's start by clearing al the basic block highlights.
    for bb_ in graph.get_vertices_data() :
        bb_.set_user_highlight(HighlightStandardColor.NoHighlightColor)

    # Get the loops
    loops = graph.detect_loops()

    # Is this bb in a loop?
    for loop in loops :
        if bb.start not in loop :
            continue
        for vertex_index in loop :
            b = graph.get_vertex_from_index(vertex_index).data
            b.set_user_highlight(HighlightStandardColor.OrangeHighlightColor)


def highlight_loop_branch (bv, address) :
    bb = bv.get_basic_blocks_at(address)[0]
    graph = graph_function_low_level_il(bb.function)

    loops = graph.detect_loops()
    dominators = graph.compute_dominators()

    # For each loop this vertex is in
    for loop in loops :
        if bb.start not in map(lambda v: graph.get_vertex_from_index(v).data[0].address, loop) :
            continue

        # Find overall dominator for this loop
        loop_dominator = find_loop_dominator(dominators, loop)

        # Get this LLILBlock
        block = graph.get_vertex_from_index(loop_dominator).data

        # Get the last instruction
        last_ins = block[-1]

        # Highlight it
        bb.function.set_user_instr_highlight(last_ins.address,
                                             HighlightStandardColor.MagentaHighlightColor)

def loop_analysis (bv) :
    report = []
    for function in bv.functions :
        graph = graph_function(function)
        report.append((function.name, len(graph.detect_loops())))

    report.sort(key=lambda x: x[1])
    report.reverse()

    markdown = "Function Name | # Detected Loops\n"
    markdown += "--- | ---\n"
    markdown += "\n".join(map(lambda x: x[0] + "|" + str(x[1]), report))

    interaction.show_markdown_report("Loop Detection", markdown)


def reaching_definitions (bv, address) :
    bb = bv.get_basic_blocks_at(address)[0]
    function = bb.function
    llil_graph = graph_function_llil_instructions(function)
    reachingDefinitions = ReachingDefinitions(llil_graph, bv).definitions

    # merge reaching definitions by address
    defaddrs = {}
    for i in reachingDefinitions :
        reachdef = reachingDefinitions[i]
        if reachdef == None :
            continue
        if reachdef.address not in defaddrs :
            defaddrs[reachdef.address] = {'live': [], 'used': [], 'defined': []}
        defaddr = defaddrs[reachdef.address]
        for l in reachdef.live :
            if l not in defaddr['live'] :
                defaddr['live'].append(l)
        for u in reachdef.used :
            if u not in defaddr['used'] :
                defaddr['used'].append(u)
        for d in reachdef.defined :
            if d not in defaddr['defined'] :
                defaddr['defined'].append(d)

    def varstring (variable) :
        return '%s@%s' % (variable.name, hex(variable.address))

    for addr in defaddrs :
        live = ','.join(map(lambda v: str(v), defaddrs[addr]['live']))
        used = ','.join(map(lambda v: str(v), defaddrs[addr]['used']))
        defined = ','.join(map(lambda v: str(v), defaddrs[addr]['defined']))
        '''
        function.set_comment(addr, 'live: %s\nused: %s\ndefined: %s' % (
            live, used, defined
        ))
        '''
        function.set_comment(addr, 'used: %s\ndefined: %s' % (
            used, defined
        ))


PluginCommand.register_for_address("Highlight Dominators",
                                   "Highlights all dominators of a block",
                                   highlight_dominators)

PluginCommand.register_for_address("Highlight Immediate Dominator",
                                   "Highlights immediate dominator of a block",
                                   highlight_immediate_dominator)

PluginCommand.register_for_address("Highlight Innermost Loop",
                                   "If this block is part of a loop, highlight the innermost loop",
                                   highlight_innermost_loop)

PluginCommand.register_for_address("Highlight Loop Branch",
                                   "If this block is part of a loop, highlight the loop's branching instruction",
                                   highlight_loop_branch)

PluginCommand.register_for_address("Highlight Predecessors",
                                   "Highlights all predecessors of a block",
                                   highlight_predecessors)

PluginCommand.register_for_address("Reaching Definitions",
                                   "Comment everything with reaching definitions",
                                   reaching_definitions)

PluginCommand.register("Loop Detection",
                       "Detect and count loops for all functions",
                       loop_analysis)
