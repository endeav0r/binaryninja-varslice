from binaryninja import *
import copy

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


def set_intersection (sets) :
    '''
    Takes a list of lists, and returns the intersection of those lists.
    '''
    if len(sets) < 1 :
        return sets
    intersection = copy.deepcopy(sets[0])
    for s in sets :
        i = 0
        while i < len(intersection) :
            if intersection[i] not in s :
                del intersection[i]
            else :
                i = i + 1
    return intersection


def set_equivalence (sets) :
    '''
    Takes a list of lists, and returns True if all lists are equivalent, False
    otherwise.
    '''
    # An empty set is equivalent
    if len(sets) < 0 :
        return True
    # Sets of differing length are obviously not equivalent
    l = len(sets[0])
    for s in sets :
        if len(s) != l :
            return False

    sets_copy = copy.deepcopy(sets)
    for i in range(len(sets_copy)) :
        sets_copy[i].sort()

    for i in range(len(sets_copy[0])) :
        for j in range(len(sets_copy)) :
            if sets_copy[0][i] != sets_copy[j][i] :
                return False

    return True


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



def edge_list_get_tail_index (edge_list, tail_index) :
    '''
    Takes a list of edges and returns an edge if the tail_index matches the
    given index, or None otherwise.
    '''
    for edge in edge_list :
        if edge.tail_index == tail_index :
            return edge
    return None



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
        @return A dict of variable identifiers to VariableAnalysis which are
                modified/written by this instruction.
        '''

        # Get an instance of our SSA creator
        ssa = getSSA()

        # TODO I wonder if this copies values, and the terrible repurcussions if
        # this doesn't... which I doubt it does... bugs galore!
        # This gives us the string identifiers of written and read registers
        written, read = il_registers(self.il)

        print written, read, self.il, self.il.operation

        # We need to first go through read registers, and see if they are in
        # our in_variables. If not, we add those.
        for r in read :
            # If this read variable doesn't exist, create it and give it a
            # unique SSA identifier
            if r not in in_variables :
                self.read[r] = VariableAnalysis(r, ssa.new_ssa(r))
            else :
                self.read[r] = in_variables[r]

        # No we go through written variables, and apply SSA to them
        written_ = {}
        for w in written :
            ww = VariableAnalysis(w, ssa.new_ssa(w))
            written_[w] = ww

        # and we apply all of our read registers to our written registers
        for w in written_ :
            for r in self.read :
                written_[w].dependencies.append(r)

        return written_




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




class Edge :
    '''
    This class represents a generic edge in a graph. It does not contain
    references to its head and tail directly, but instead indicies to the head
    and tail.

    You should not store references to this edge directly.
    '''
    def __init__ (self, graph, index, head_index, tail_index, data=None) :
        '''
        Create an edge. You should not call this directly. Call
        graph.add_edge() instead.
        '''
        self.graph = graph
        self.index = index
        self.head_index = head_index
        self.tail_index = tail_index
        self.data = data

    def head (self) :
        '''
        Returns a reference to the head vertex of this edge.
        '''
        return self.graph.vertex_from_index(self.head_index)

    def tail (self) :
        '''
        Returns a reference to the tail vertex of this edge.
        '''
        return self.graph.vertex_from_index(self.tail_index)



class Vertex :
    '''
    This class represents a generic vertex in a graph.
    '''
    def __init__ (self, graph, index, data=None) :
        '''
        Creates a vertex. You should not call this directly. Call
        graph.add_vertex() instead.
        '''
        self.graph = graph
        self.index = index
        self.data = data

    def get_predecessor_indices (self) :
        predecessor_edges = self.graph.get_edges_by_tail_index(self.index)
        return map(lambda e: e.head_index, predecessor_edges)

    def get_predecessors (self) :
        return map(lambda i: self.graph.get_vertex_from_index(i),
                   self.get_predecessor_indices())
        
    def get_successor_indices (self) :
        successor_edges = self.graph.get_edges_by_head_index(self.index)
        return map(lambda e: e.tail_index, successor_edges)

    def get_successors (self) :
        return map(lambda i: self.graph.get_vertex_from_index(i),
                   self.get_successor_indices())


class Graph :

    def __init__ (self, entry_index=None) :
        # When we create vertices, if an index is not specified, we increment
        # this to ensure we are creating unique vertex indicies
        self.next_vertex_index = -1000
        # A mapping of vertices by vertex index to vertex
        self.vertices = {}
        # When we create edges, we increment this to create unique edge indicies
        self.next_edge_index = 1

        # We keep references to the same edge in three different places to speed
        # up the searching for edges

        # A mapping of edges by edge index to edge
        self.edges = {}
        # A mapping of edges by head_index to edge
        self.edges_by_head_index = {}
        # A mapping of edges by tail_index to edge
        self.edges_by_tail_index = {}

        # An entry_index simplifies lots of stuff, like computing dominators
        self.entry_index = entry_index

    def add_edge (self, head, tail, data=None) :
        '''
        Adds an edge to the graph by giving references to the head and tail
        vertices.
        This is just a wrapper for add_edge_by_indices.
        '''
        return self.add_edge_by_indices(head.index, tail.index, data)

    def add_edge_by_indices (self, head_index, tail_index, data=None) :
        '''
        Adds an edge to the graph. Will fail if:
        1) There is no vertex in the graph for head_index.
        2) There is no vertex in the graph for tail_index.
        3) An edge already exists from head -> tail.

        @param head_index The index of the head vertex.
        @param tail_index The index of the tail vertex.
        @param data Any data you would like associated with this edge.
        @return A reference to the new edge if it was created, or None on
                failure.
        '''

        # Ensure we have a valid head and tail
        if not self.vertices.has_key(head_index) :
            return None
        if not self.vertices.has_key(tail_index) :
            return None

        # If we already have an edge here, don't add a new one
        if     self.edges_by_head_index.has_key(head_index) \
           and edge_list_get_tail_index(self.edges_by_head_index[head_index], tail_index) :
            return None

        # Create our new edge
        index = self.next_edge_index
        edge = Edge(self, index, head_index, tail_index, data)

        # Add it to our dict of edges
        self.edges[index] = edge

        # Add this edge to our lists of edges by head_index and tail_index
        if not self.edges_by_head_index.has_key(head_index) :
            self.edges_by_head_index[head_index] = [edge]
        else :
            self.edges_by_head_index[head_index].append(edge)

        if not self.edges_by_tail_index.has_key(tail_index) :
            self.edges_by_tail_index[tail_index] = [edge]
        else :
            self.edges_by_tail_index[tail_index].append(edge)

        # Return the edge
        return edge

    def add_vertex (self, index=None, data=None) :
        '''
        Adds a vertex to the graph. Index represents a desired index for this
        vertex, such as an address in a CFG, and data represents data you would
        like to associate with this vertex. If no index is given, one will be
        assigned.

        @param index A desired index for this vertex
        @param data Data you would like to associate with this vertex
        @return The newly created vertex, or None if the vertex could not be
                created.
        '''
        if index == None :
            index = self.next_vertex_index
            self.next_vertex_index += 1
            while self.vertices.has_key(index) :
                index = self.next_vertex_index
                self.next_vertex_index += 1
        else :
            if self.vertices.has_key(index) :
                return None
        self.vertices[index] = Vertex(self, index, data)
        return self.vertices[index]

    def compute_dominators (self) :
        '''
        Returns a mapping of vertex nodes to a list of dominators for that
        vertex. This is quadratic time, so have fun!.
        '''
        dominators = {}

        for vertex_index in self.vertices :
            dominators[vertex_index] = self.vertices.keys()

        dominators_changed = True

        while dominators_changed :
            dominators_changed = False

            # For each vertex
            keys = dominators.keys()
            for vertex_index in keys :
                vertex = self.get_vertex_from_index(vertex_index)
                # Recompute dominators
                doms = set_intersection([dominators[i] for i in vertex.get_predecessor_indices()])
                doms += [vertex_index]
                # Is this new set of dominators different from what we had before ?
                if set_equivalence([doms, dominators[vertex_index]]) :
                    continue
                dominators_changed = True
                dominators[vertex_index] = doms

        return dominators


    def compute_predecessors (self) :
        '''
        Returns a mapping of a vertex index to a list of vertex indices, where
        the key is given vertex and the value is a list of all vertices which
        are predecessors to that vertex.
        '''

        # Set our initial predecessors for each vertex
        predecessors = {}
        for vertex_index in self.vertices :
            vertex = self.vertices[vertex_index]
            predecessors[vertex_index] = vertex.get_predecessor_indices()

        # We now do successive propogation passes until we no longer propogate
        propogate = True
        while propogate :
            propogate = False
            # For each vertex in the graph
            for vertex_index in predecessors :
                # For each predecessor of this vertex
                for predecessor_index in predecessors[vertex_index] :
                    # Ensure all of these predecessor's are predecessors of this
                    # vertex
                    for pp_index in predecessors[predecessor_index] :
                        if pp_index not in predecessors[vertex_index] :
                            predecessors[vertex_index].append(pp_index)
                            propogate = True

        return predecessors

    def get_edges_by_head_index (self, head_index) :
        '''
        Returns all edges who have a given head index. This is the same as the
        successor edges for a vertex by index.

        @param head_index The index of the vertex.
        @return A list of all edges with a head_index of head_index. An empty
                list will be returned if no such edges exist, including the case
                where a vertex with index head_index does not exist.
        '''
        if not self.edges_by_head_index.has_key(head_index) :
            return []
        return self.edges_by_head_index[head_index]

    def get_edges_by_tail_index (self, tail_index) :
        '''
        Returns all edges who have a given tail index. This is the same as the
        predecessor edges for a vertex by index.

        @param tail_index The index of the vertex.
        @return A list of all edges with a tail_index of tail_index. An empty
                list will be returned if no such edges exist, including the case
                where a vertex with index tail_index does not exist.
        '''
        if not self.edges_by_tail_index.has_key(tail_index) :
            return []
        return self.edges_by_tail_index[tail_index]

    def get_vertex_from_index (self, index) :
        '''
        Returns a vertex with the given index.

        @param index The index of the vertex to retrieve.
        @return The vertex, or None if the vertex does not exist.
        '''
        if not self.vertices.has_key(index) :
            return None
        return self.vertices[index]

    def get_vertices_data (self) :
        return map(lambda x: x.data, [self.vertices[y] for y in self.vertices])


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

    log.log_warn(dominators)

    # Highlight all predecessors blue.
    for dominator in dominators :
        bb = graph.get_vertex_from_index(dominator).data
        bb.set_user_highlight(HighlightStandardColor.GreenHighlightColor)




PluginCommand.register_for_address("Highlight Predecessors",
                                   "Highlights all predecessors of a block",
                                   highlight_predecessors)
PluginCommand.register_for_address("Highlight Dominators",
                                   "Highlights all dominators of a block",
                                   highlight_dominators)