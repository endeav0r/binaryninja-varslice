from binaryninja import *
import copy

from graph import Graph, find_loop_dominator

SSAObject = None

class SSA :
    '''
    This object tracks unique SSA-identifiers to apply SSA-logic to variables
    '''
    def __init__ (self) :
        self.variables = {}

    def new_ssa (self, variable_name) :
        '''
        Creates a new unique SSA-identifier for a variable and returns it

        @param variable_name The name of the variable we want a unique
                             SSA-identifier for.
        @return A unique SSA-identifier for the variable
        '''
        if not self.variables.has_key(variable_name) :
            self.variables[variable_name] = 1
        else :
            self.variables[variable_name] += 1
        return self.variables[variable_name]


SSAObject = None


def getSSA() :
    '''
    Implements a SSA singleton
    '''
    global SSAObject
    if SSAObject == None :
        SSAObject = SSA()
    return SSAObject


def expression_registers (il) :
    '''
    Returns a list of registers which are used by this expression.
    '''
    if il.operation in (LowLevelILOperation.LLIL_ADD,
                        LowLevelILOperation.LLIL_ADC,
                        LowLevelILOperation.LLIL_SUB,
                        LowLevelILOperation.LLIL_SBB,
                        LowLevelILOperation.LLIL_AND,
                        LowLevelILOperation.LLIL_OR,
                        LowLevelILOperation.LLIL_XOR,
                        LowLevelILOperation.LLIL_LSL,
                        LowLevelILOperation.LLIL_LSR,
                        LowLevelILOperation.LLIL_ASR,
                        LowLevelILOperation.LLIL_ROL,
                        LowLevelILOperation.LLIL_RLC,
                        LowLevelILOperation.LLIL_ROR,
                        LowLevelILOperation.LLIL_RRC,
                        LowLevelILOperation.LLIL_MUL,
                        LowLevelILOperation.LLIL_MULU_DP,
                        LowLevelILOperation.LLIL_MULS_DP,
                        LowLevelILOperation.LLIL_DIVU,
                        LowLevelILOperation.LLIL_DIVS,
                        LowLevelILOperation.LLIL_MODU,
                        LowLevelILOperation.LLIL_MODS) :
        return expression_registers(il.left) + expression_registers(il.right)

    if il.operation in (LowLevelILOperation.LLIL_DIVS_DP,
                        LowLevelILOperation.LLIL_DIVU_DP,
                        LowLevelILOperation.LLIL_MODU_DP,
                        LowLevelILOperation.LLIL_MODS_DP) :
        # TODO this is an unhandled case ! I don't know what to do here
        return expression_registers(il.lo) + expression_registers(il.hi) + expression_registers(il.right)

    if il.operation == LowLevelILOperation.LLIL_LOAD :
        return expression_registers(il.src)

    if il.operation == LowLevelILOperation.LLIL_REG :
        return [il.src]

    if il.operation == LowLevelILOperation.LLIL_CONST :
        return []

    if il.operation == LowLevelILOperation.LLIL_FLAG :
        return [il.src]

    if il.operation in (LowLevelILOperation.LLIL_NEG,
                        LowLevelILOperation.LLIL_NOT,
                        LowLevelILOperation.LLIL_SX,
                        LowLevelILOperation.LLIL_ZX) :
        return expression_registers(il.src)

    '''
    if il.operation == LowLevelILOperation.LLIL_IF :
        return expression_registers(il.condition)
    '''

    if il.operation == LowLevelILOperation.LLIL_FLAG_COND :
        return expression_registers(il.condition)

    if il.operation in (LowLevelILOperation.LLIL_CMP_E,
                        LowLevelILOperation.LLIL_CMP_NE,
                        LowLevelILOperation.LLIL_CMP_SLT,
                        LowLevelILOperation.LLIL_CMP_ULT,
                        LowLevelILOperation.LLIL_CMP_SLE,
                        LowLevelILOperation.LLIL_CMP_ULE,
                        LowLevelILOperation.LLIL_CMP_SGE,
                        LowLevelILOperation.LLIL_CMP_UGE,
                        LowLevelILOperation.LLIL_CMP_SGT,
                        LowLevelILOperation.LLIL_CMP_UGT,
                        LowLevelILOperation.LLIL_TEST_BIT) :
        return expression_registers(il.left) + expression_registers(il.right)

    if il.operation == LowLevelILOperation.LLIL_BOOL_TO_INT :
        return expression_registers(il.src)

    if il.operation == LowLevelILOperation.LLIL_POP :
        return []

    print 'error', il.operation, il


def il_registers (il) :

    '''
    Returns a tuple of written,read registers used by this LLIL instruction.
    '''
    written = []
    read = []
    if il.operation in (LowLevelILOperation.LLIL_NOP,
                        LowLevelILOperation.LLIL_POP,
                        LowLevelILOperation.LLIL_NORET,
                        LowLevelILOperation.LLIL_GOTO) :
        return [], []
        pass
    elif il.operation == LowLevelILOperation.LLIL_IF :
        read = expression_registers(il.condition)
    elif il.operation == LowLevelILOperation.LLIL_SET_REG :
        written.append(il.dest)
        read = expression_registers(il.src)

    elif il.operation == LowLevelILOperation.LLIL_SET_REG_SPLIT :
        written.append(il.hi)
        written.append(il.lo)
        read = expression_registers(il.src)

    elif il.operation == LowLevelILOperation.LLIL_SET_FLAG :
        written.append(il.flag)
        read = expression_registers(il.src)

    elif il.operation == LowLevelILOperation.LLIL_STORE :
        written = expression_registers(il.dest)
        read = expression_registers(il.src)

    elif il.operation == LowLevelILOperation.LLIL_PUSH :
        read = expression_registers(il.src)

    elif il.operation == LowLevelILOperation.LLIL_JUMP :
        read = expression_registers(il.dest)

    elif il.operation == LowLevelILOperation.LLIL_JUMP_TO :
        read = expression_registers(il.dest)

    elif il.operation == LowLevelILOperation.LLIL_CALL :
        read = expression_registers(il.dest)

    elif il.operation == LowLevelILOperation.LLIL_RET :
        read = expression_registers(il.dest) # architectures with link registers

    elif il.operation == LowLevelILOperation.LLIL_SYSCALL :
        # TODO syscall is unhandled... probably impossible to handle
        pass

    elif il.operation == LowLevelILOperation.LLIL_BP :
        pass

    elif il.operation == LowLevelILOperation.LLIL_TRAP :
        pass

    elif il.operation in (LowLevelILOperation.LLIL_UNDEF,
                          LowLevelILOperation.LLIL_UNIMPL,
                          LowLevelILOperation.LLIL_UNIMPL_MEM) :
        pass
    else :
        print "unhandled top-level instruction"
        print il
        print il.operation
        raise

    return written, read



class LLILAnalysis :
    '''
    This is a wrapper around LLILInstruction we can perform analysis around
    '''
    def __init__ (self, llil) :
        self.llil = llil
        self.written_registers = []
        self.read_registers = []

    def analyze (self) :
        written, read = il_registers(self.llil)
        self.written_registers = written
        self.read_registers = read



class VariableAnalysis :
    '''
    This is a placeholder for strings with integer SSA-identifiers.
    '''
    def __init__ (self, identifier, ssa) :
        '''
        This is a string identifier for the variable, the most basic identifier
        for the variable. For example, "eax".
        '''
        self.identifier = identifier

        '''
        This is a unique SSA-identifier for the variable, which makes it unique
        amongst the analysis. For example, if this is the 4th eax which is
        assigned, this will be 4. Totally variable, just needs to be unique.
        '''
        self.ssa = ssa

        '''
        This list contains references to other VariableAnalysis instances,
        unique by identifier, to variables that influence this variable.
        '''
        self.dependencies = []

    def str (self) :
        return '%s_%d' % (self.identifier, self.ssa)

    def __str__ (self) :
        return self.str()


class InsAnalysis :
    '''
    A meta-instruction which we do analysis over
    '''
    def __init__ (self, basic_block, bb_index, il) :
        '''
        Constructor for InsAnalysis

        @param basic_block The BBAnalysis this InsAnalysis belongs to.
        @param bb_index The index in BBAnalysis of this instruction.
        '''
        self.basic_block = basic_block
        self.bb_index = bb_index
        self.il = il
        self.written = {}
        self.read = {}

    def apply_ssa (self, in_variables) :
        '''
        Takes a dict of identifiers to VariableAnalysis instances, and returns
        a dict of identifiers to VariableAnalysis instances of variables which
        are modified by in_variables.

        @param in_variables a dict of variable identifiers to VariableAnalysis
                            which are valid before this instruction is executed.
        @return A dict of identifiers to VariableAnalysis instances, which are
                the state of all variables after this instruction. I.E. this is
                in_variables with identifiers that were written replaced with
                their new VariableAnalysis instances.
        '''

        # Get an instance of our SSA creator
        ssa = getSSA()

        written, read = il_registers(self.il)

        print written, read, self.il, self.il.operation

        # We don't want to modify the in_variables we were given. We can now
        # changes this up at will.
        in_variables = copy.deepcopy(in_variables)

        # We need to first go through read registers, and see if they are in
        # our in_variables. If not, we add those.
        for r in read :
            # If this read variable doesn't exist, create it and give it a
            # unique SSA identifier
            if r not in in_variables :
                self.read[r] = VariableAnalysis(r, ssa.new_ssa(r))
            else :
                self.read[r] = copy.deepcopy(in_variables[r])

        # No we go through written variables, and apply SSA to them
        written_ = {}
        for w in written :
            ww = VariableAnalysis(w, ssa.new_ssa(w))
            written_[w] = ww

        # and we apply all of our read registers to our written registers
        for w in written_ :
            for r in self.read :
                written_[w].dependencies.append(copy.deepcopy(r))

        # save our written registers
        self.written = copy.deepcopy(written_)

        # now overwrite values in in_variables to create our result
        # written_ shouldn't have any references anywhere and should be a pure
        # copy.
        for w in written_ :
            in_variables[w.identifier] = w

        return in_variables # in is the new out




class BBAnalysis :
    '''
    This is a wrapper around a binary ninja basic block. We use this to track
    analysis around this block when creating a vertex in our graph.
    '''
    def __init__ (self, basic_block) :
        self.basic_block = basic_block
        self.instructions = []
        for i in range(len(self.basic_block)) :
            self.instructions.append(InsAnalysis(self, i, self.basic_block[i]))

    def print_il_instructions (self) :
        for ins in self.basic_block :
            print ins.operation, ins

    def read_written_registers (self) :
        written_ = []
        read_ = []
        for il in self.basic_block :
            written, read = il_registers(il)
            print written, read
            for r in read :
                if r not in written_ :
                    read_.append(r)
            for w in written :
                written_.append(w)
        return read_, written_

    def apply_ssa (self, in_variables) :
        variables = in_variables
        for i in range(len(self.basic_block)) :
            out_variables = self.instructions[i].apply_ssa(variables)
            for k in out_variables :
                variables[k] = out_variables[k]
        return variables





def graph_function_low_level_il (function) :
    graph = Graph(0)

    # get the low_level_il basic blocks
    basic_blocks = function.low_level_il.basic_blocks

    # We are going to add each basic block to our graph
    for basic_block in basic_blocks :
        graph.add_vertex(basic_block.start, BBAnalysis(basic_block))

    # Now we are going to add all the edges
    for basic_block in basic_blocks :
        for outgoing_edge in basic_block.outgoing_edges :
            target = outgoing_edge.target
            graph.add_edge_by_indices(basic_block.start, target.start, None)

    # Now return the graph
    return graph


def graph_function (function) :
    graph = Graph(function.start)

    basic_blocks = function.basic_blocks

    for basic_block in basic_blocks :
        graph.add_vertex(basic_block.start, basic_block)

    for basic_block in basic_blocks :
        for outgoing_edge in basic_block.outgoing_edges :
            target = outgoing_edge.target
            graph.add_edge_by_indices(basic_block.start, target.start, None)

    return graph


class ProcessGraph :
    '''
    Helper class for the process_graph function
    '''
    def __init__ (self, vertex, in_variables) :
        self.vertex = vertex
        self.in_variables = in_variables


def process_graph (graph, start_vertex_index) :
    # A set of vertex indicies we've processed
    processed_vertix_indicies = []

    # vertex will be the current vertex we're processing
    vertex = graph.get_vertex_from_index(start_vertex_index)

    # and vertex_stack will maintain the vertices we need to search
    pg_stack = []
    pg_stack.append(ProcessGraph(vertex, {}))

    # a list of vertices we've already searched, by their .start value
    processed = []

    while len(pg_stack) > 0 :
        # pop the first vertex off the stack
        processGraph = pg_stack[0]
        pg_stack = pg_stack[1:]

        processed.append(processGraph.vertex.data.basic_block.start)

        # Get this vertex's basic block
        bb = processGraph.vertex.data

        print 'processing', bb.basic_block.start

        # And get this vertex's in_variables
        in_variables = processGraph.in_variables

        # With these in_variables, apply SSA over this block and receive this
        # blocks out_variables
        out_variables = bb.apply_ssa(in_variables)
        print "out_variables", out_variables

        for oe in bb.basic_block.outgoing_edges :
            if oe.target.start not in processed :
                pg_stack.append(ProcessGraph(graph.get_vertex_from_index(oe.target.start), in_variables))
            else :
                # TODO handle loops
                pass


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
        if bb.start not in map(lambda v: graph.get_vertex_from_index(v).data.basic_block[0].address, loop) :
            continue

        # Find overall dominator for this loop
        loop_dominator = find_loop_dominator(dominators, loop)

        # Get this LLILBlock
        block = graph.get_vertex_from_index(loop_dominator).data.basic_block

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

PluginCommand.register("Loop Detection",
                       "Detect and count loops for all functions",
                       loop_analysis)
