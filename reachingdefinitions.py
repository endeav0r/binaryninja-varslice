import analysis
import copy

'''
We will track the state of each instruction, where the state will be the
definitions of all live variables at the conclusion of the execution of the
instruction.

There can be only one live instance of an instruction at the conclusion of an
instruction, though there can be multiple live instances of an instruction prior
to the execution of an instruction.
'''


class Variable :
    def __init__ (self, name, address=None) :
        self.address = address
        self.name = name

    def __eq__ (self, other) :
        if isinstance(other, Variable) :
            if self.address == other.address and self.name == other.name :
                return True
        return False

    def __ne__ (self, other) :
        return not self.__eq__(other)

    def __hash__ (self) :
        if self.address == None :
            return self.name.__hash__()
        return (self.name.__hash__() << 32) + self.address

    def __repr__ (self) :
        if self.address == None :
            return '(%s@not_live)' % (self.name)
        return '(%s@%s)' % (self.name, hex(self.address))


class ReachingDefinition :
    def __init__ (self, llil, live=[]) :
        self.address = llil.address
        self.live = live
        self.used = []
        self.defined = []

    def set_live (self, live) :
        self.live = live

    def set_used (self, used) :
        self.used = []
        '''
        When setting used variables, we check to see if they are live, and if
        they are we create a used instance for each live instance
        '''
        for u in used :
            is_live = False
            for l in self.live :
                if u.name == l.name :
                    is_live = True
                    self.used.append(copy.deepcopy(l))
            if not is_live :
                self.used.append(copy.deepcopy(u))

    def add_defined (self, defined) :
        '''
        Takes a single Variable instance, removes all prior instances of this
        variable from live, then adds it to both live and defined
        '''
        i = 0
        while i < len(self.live) :
            if self.live[i].name == defined.name :
                del self.live[i]
            else :
                i += 1
        self.live.append(copy.deepcopy(defined))
        self.defined.append(copy.deepcopy(defined))

    def set_defined (self, defined) :
        '''
        Takes a list of defined Variable instances and calls self.add_defined
        over them
        '''
        for d in defined :
            self.add_defined(d)

    def merge (self, other) :
        '''
        Used to merge results from multiple expressions in the same instruction
        '''
        if other == None :
            return

        for l in other.live :
            if l not in self.live :
                self.live.append(copy.deepcopy(l))

        for l in other.used :
            if u not in self.used :
                self.used.append(copy.deepcopy(u))

        for d in other.defined :
            if d not in self.defined :
                self.add_defined(d)

    def __eq__ (self, other) :
        if other == None :
            return False

        if set(self.live) != set(other.live) :
            print 'live_diff', self.live, other.live
            return False
        elif set(self.used) != set(other.used) :
            print 'used_diff', self.used, other.used
            return False
        elif set(self.defined) != set(other.defined) :
            print 'defined_diff', self.defined, other.defined
            return False
        return True

    def __ne__ (self, other) :
        return not self.__eq__(other)


class ReachingDefinitions (analysis.AnalysisModel) :

    def __init__ (self, llil_instructions_graph, bv) :
        super(ReachingDefinitions, self).__init__()
        self.llil_handler_print = True
        self.bv = bv

        '''
        Mapping of addresses to RDInstruction
        '''
        self.defs = {}

        self.definitions = self.fixpoint_forward(llil_instructions_graph)


    def prepare_op (self, llil, data) :
        reachingDefinition = ReachingDefinition(llil)
        if data != None :
            reachingDefinition.set_live(data.live)
        return reachingDefinition


    def join_lhs_rhs (self, llil, lhs, rhs) :
        reachingDefinition = ReachingDefinition(llil)

        reachingDefinition.set_live(copy.deepcopy(lhs.live))

        for l in rhs.live :
            if l not in reachingDefinition.live :
                reachingDefinition.live.append(copy.deepcopy(l))

        return reachingDefinition


    def state_equivalence_lhs_rhs (self, lhs, rhs) :
        # We will compare keys between RDInstruction states first
        if set(lhs.keys()) != set(rhs.keys()) :
            return False

        # Now we will compare the values of all variables in each state
        for key in lhs :
            if set(lhs[key]) != set(rhs[key]) :
                return False

        return True


    def reg_name (self, reg_name) :
        return self.bv.arch.regs[reg_name].full_width_reg


    def _arith (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _arith', llil
        definitions = self.optable[llil.left.operation](llil.left, data)
        definitions += self.optable[llil.right.operation](llil.right, data)
        return definitions

    def _arith_db (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _arith_dp', llil

    def _unary (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _unary', llil

    def _set_reg (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _set_reg', llil
        reachingDefinition = self.prepare_op(llil, data)
        reachingDefinition.set_used(self.recursive_op(llil.src))
        defined = [Variable(self.reg_name(llil.dest), llil.address)]
        reachingDefinition.set_defined(defined)
        return reachingDefinition

    def _cmp (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _cmp', llil
        variables = self.recursive_op(llil.left)
        variables += self.recursive_op(llil.right)
        return variables

    def _reg (self, llil, data=None) :
        return [Variable(self.reg_name(llil.src))]

    def _const (self, llil, data=None) :
        return []

    def _flag_cond (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _flag_cond', llil

    def _flag (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _flag', llil

    def _load (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _load', llil
        return self.recursive_op(llil.src)

    def _store (self, llil, data=None) :
        reachingDefinition = self.prepare_op(llil, data)
        if self.llil_handler_print :
            print ' _store', llil
        variables = self.recursive_op(llil.dest)
        variables += self.recursive_op(llil.src)
        reachingDefinition.set_used(variables)
        return reachingDefinition

    def _push (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _push',
        reachingDefinition = self.prepare_op(llil, data)
        reachingDefinition.set_used(self.recursive_op(llil.src))
        return reachingDefinition

    def _pop (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _pop', llil
        return [] # looks like pop is a rhs

    def _noret (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _noret', llil

    def _goto (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _goto', llil
        return self.prepare_op(llil, data)

    def _If (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _If', llil
        reachingDefinition = self.prepare_op(llil, data)
        reachingDefinition.set_used(self.recursive_op(llil.condition))
        return reachingDefinition

    def _bool_to_int (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _bool_to_int', llil

    def _jump (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _jump', llil

    def _jump_to (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _jump_to', llil

    def _call (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _call'
        reachingDefinition = self.prepare_op(llil, data)
        reachingDefinition.set_used(self.recursive_op(llil.dest))
        return reachingDefinition


    def _ret (self, llil, data=None) :
        if self.llil_handler_print :
            print ' _ret', llil
        return None # No need to return anything here

    def _test_bit (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _test_bit', llil

    def _syscall (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _syscall', llil

    def _bp (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _bp', llil

    def _trap (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _trap', llil

    def _undef (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _undef', llil

    def _unimpl (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _unimpl'

    def _unimpl_mem (self, llil, data=None) :
        if self.llil_handler_print :
            print ' UNHANDLED _unimpl_mem'
