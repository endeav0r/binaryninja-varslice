import sys

EXAMPLE_APPLICATION_BINARY = "/Users/endeavor/code/complexity/example0"
BINARY_NINJA_API_PATH = "/Applications/Binary Ninja.app/Contents/Resources/python"

sys.path.append(BINARY_NINJA_API_PATH)

from binaryninja import *

SSAObject = None

class SSA :
	'''
	This object tracks unique SSA-identifiers to apply SSA-logic to variables
	'''
	def __init__ (self) :
		self.variables = {}

	def make_ssa (self, variable_name) :
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
	if    il.operation == LowLevelILOperation.LLIL_ADD \
	   or il.operation == LowLevelILOperation.LLIL_ADC \
	   or il.operation == LowLevelILOperation.LLIL_SUB \
	   or il.operation == LowLevelILOperation.LLIL_SBB \
	   or il.operation == LowLevelILOperation.LLIL_AND \
	   or il.operation == LowLevelILOperation.LLIL_OR \
	   or il.operation == LowLevelILOperation.LLIL_XOR \
	   or il.operation == LowLevelILOperation.LLIL_LSL \
	   or il.operation == LowLevelILOperation.LLIL_LSR \
	   or il.operation == LowLevelILOperation.LLIL_ASR \
	   or il.operation == LowLevelILOperation.LLIL_ROL \
	   or il.operation == LowLevelILOperation.LLIL_RLC \
	   or il.operation == LowLevelILOperation.LLIL_ROR \
	   or il.operation == LowLevelILOperation.LLIL_RRC \
	   or il.operation == LowLevelILOperation.LLIL_MUL \
	   or il.operation == LowLevelILOperation.LLIL_MULU_DP \
	   or il.operation == LowLevelILOperation.LLIL_MULS_DP \
	   or il.operation == LowLevelILOperation.LLIL_DIVU \
	   or il.operation == LowLevelILOperation.LLIL_DIVS \
	   or il.operation == LowLevelILOperation.LLIL_MODU \
	   or il.operation == LowLevelILOperation.LLIL_MODS :
	    print il.operation
	    return expression_registers(il.left) + expression_registers(il.right)

	if    il.operation == LowLevelILOperation.LLIL_DIVS_DP \
	   or il.operation == LowLevelILOperation.LLIL_DIVU_DP \
	   or il.operation == LowLevelILOperation.LLIL_MODU_DP \
	   or il.operation == LowLevelILOperation.LLIL_MODS_DP :
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

	if    il.operation == LowLevelILOperation.LLIL_NEG \
	   or il.operation == LowLevelILOperation.LLIL_NOT \
	   or il.operation == LowLevelILOperation.LLIL_SX \
	   or il.operation == LowLevelILOperation.LLIL_ZX :
	    return expression_registers(il.src)

	if il.operation == LowLevelILOperation.LLIL_IF :
		return expression_registers(il.condition)

	if il.operation == LowLevelILOperation.LLIL_FLAG_COND :
		return expression_registers(il.condition)

	if    il.operation == LowLevelILOperation.LLIL_CMP_E \
	   or il.operation == LowLevelILOperation.LLIL_CMP_NE \
	   or il.operation == LowLevelILOperation.LLIL_CMP_SLT \
	   or il.operation == LowLevelILOperation.LLIL_CMP_ULT \
	   or il.operation == LowLevelILOperation.LLIL_CMP_SLE \
	   or il.operation == LowLevelILOperation.LLIL_CMP_ULE \
	   or il.operation == LowLevelILOperation.LLIL_CMP_SGE \
	   or il.operation == LowLevelILOperation.LLIL_CMP_UGE \
	   or il.operation == LowLevelILOperation.LLIL_CMP_SGT \
	   or il.operation == LowLevelILOperation.LLIL_CMP_UGT \
	   or il.operation == LowLevelILOperation.LLIL_TEST_BIT :
	    return expression_registers(il.lhs) + expression_registers(il.rhs)

	if il.operation == LowLevelILOperation.LLIL_BOOL_TO_INT :
		return expression_registers(il.src)


def il_registers (il) :
	'''
	Returns a tuple of written,read registers used by this LLIL instruction.
	'''
	written = []
	read = []
	if    il.operation == LowLevelILOperation.LLIL_NOP \
	   or il.operation == LowLevelILOperation.LLIL_POP \
	   or il.operation == LowLevelILOperation.LLIL_NORET \
	   or il.operation == LowLevelILOperation.LLIL_GOTO :
		pass

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

	elif    il.operation == LowLevelILOperation.UNDEF \
		 or il.operation == LowLevelILOperation.UNIMPL \
		 or il.operation == LowLevelILOperation.UNIMPL_MEM :
		pass

	else :
		print "unhandled top-level instruction"
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


class WRIns :
	'''
	A meta-instruction which tracks SSA instructions for an LLIL instruction.
	'''
	def __init__ (self, written, read) :
		self.written = written
		self.read = read


class BBAnalysis :
	'''
	This is a wrapper around a binary ninja basic block. We use this to track
	analysis around this block when creating a vertex in our graph.
	'''
	def __init__ (self, basic_block) :
		self.basic_block = basic_block

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

class Graph :
	def __init__ (self) :
		self.next_vertex_index = 1
		self.vertices = {}
		self.next_edge_index = 1
		self.edges = {}
		self.edges_by_head_index = {}
		self.edges_by_tail_index = {}

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

	def edges_by_head_index (self, head_index) :
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

	def edges_by_tail_index (self, tail_index) :
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
		pass

	def get_vertex_from_index (self, index) :
		'''
		Returns a vertex with the given index.

		@param index The index of the vertex to retrieve.
		@return The vertex, or None if the vertex does not exist.
		'''
		if not self.vertices.has_key(index) :
			return None
		return self.vertices[index]


def graph_function (function) :
	graph = Graph()

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

bvt = BinaryViewType['Mach-O']
bv = bvt.open(EXAMPLE_APPLICATION_BINARY)
print bvt
print bv
bv.update_analysis_and_wait()
bb = bv.get_basic_blocks_at(0x100000f2e)[0]
print bb
print bb.function

graph = graph_function(bb.function)
print graph.vertices

bb0 = graph.get_vertex_from_index(0).data

print bb0.read_written_registers()



def instruction_slice (bv, address):
    log.log(enums.LogLevel.InfoLog, '%x' % address)

PluginCommand.register_for_address("Instruction Slice",
                                   "Show instruction dependencies for instruction",
								   instruction_slice)
