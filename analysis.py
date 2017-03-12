from binaryninja import *
import copy

class AnalysisModel (object) :

    def __init__ (self) :
        self.optable = {
            LowLevelILOperation.LLIL_ADD : self._add,
            LowLevelILOperation.LLIL_ADC : self._adc,
            LowLevelILOperation.LLIL_SUB : self._sub,
            LowLevelILOperation.LLIL_SBB : self._sbb,
            LowLevelILOperation.LLIL_AND : self._and,
            LowLevelILOperation.LLIL_OR  : self._or,
            LowLevelILOperation.LLIL_XOR : self._xor,
            LowLevelILOperation.LLIL_LSL : self._lsl,
            LowLevelILOperation.LLIL_LSR : self._lsr,
            LowLevelILOperation.LLIL_ASR : self._asr,
            LowLevelILOperation.LLIL_ROL : self._rol,
            LowLevelILOperation.LLIL_RLC : self._rlc,
            LowLevelILOperation.LLIL_ROR : self._ror,
            LowLevelILOperation.LLIL_RRC : self._rrc,
            LowLevelILOperation.LLIL_MUL : self._mul,
            LowLevelILOperation.LLIL_DIVU : self._divu,
            LowLevelILOperation.LLIL_DIVS : self._divs,
            LowLevelILOperation.LLIL_MODU : self._modu,
            LowLevelILOperation.LLIL_MODS : self._mods,

            LowLevelILOperation.LLIL_MULU_DP : self._mulu_dp,
            LowLevelILOperation.LLIL_MULS_DP : self._muls_dp,
            LowLevelILOperation.LLIL_DIVU_DP : self._divu_dp,
            LowLevelILOperation.LLIL_DIVS_DP : self._divs_dp,
            LowLevelILOperation.LLIL_MODU_DP : self._modu_dp,
            LowLevelILOperation.LLIL_MODS_DP : self._mods_dp,

            LowLevelILOperation.LLIL_NEG : self._neg,
            LowLevelILOperation.LLIL_NOT : self._not,
            LowLevelILOperation.LLIL_SX : self._sx,
            LowLevelILOperation.LLIL_ZX : self._zx,
            LowLevelILOperation.LLIL_SET_REG : self._set_reg,
            LowLevelILOperation.LLIL_SET_REG_SPLIT : self._set_reg_split,

            LowLevelILOperation.LLIL_CMP_E : self._cmp_e,
            LowLevelILOperation.LLIL_CMP_NE : self._cmp_ne,
            LowLevelILOperation.LLIL_CMP_SLT : self._cmp_slt,
            LowLevelILOperation.LLIL_CMP_ULT : self._cmp_ult,
            LowLevelILOperation.LLIL_CMP_SLE : self._cmp_sle,
            LowLevelILOperation.LLIL_CMP_ULE : self._cmp_ule,
            LowLevelILOperation.LLIL_CMP_SGE : self._cmp_sge,
            LowLevelILOperation.LLIL_CMP_UGE : self._cmp_uge,
            LowLevelILOperation.LLIL_CMP_SGT : self._cmp_sgt,
            LowLevelILOperation.LLIL_CMP_UGT : self._cmp_ugt,

            LowLevelILOperation.LLIL_REG : self._reg,
            LowLevelILOperation.LLIL_CONST : self._const,
            LowLevelILOperation.LLIL_FLAG_COND : self._flag_cond,
            LowLevelILOperation.LLIL_FLAG : self._flag,

            LowLevelILOperation.LLIL_LOAD : self._load,
            LowLevelILOperation.LLIL_STORE : self._store,
            LowLevelILOperation.LLIL_PUSH : self._push,
            LowLevelILOperation.LLIL_POP : self._pop,

            LowLevelILOperation.LLIL_NORET : self._noret,
            LowLevelILOperation.LLIL_GOTO : self._goto,
            LowLevelILOperation.LLIL_IF : self._If,
            LowLevelILOperation.LLIL_BOOL_TO_INT : self._bool_to_int,
            LowLevelILOperation.LLIL_JUMP : self._jump,
            LowLevelILOperation.LLIL_JUMP_TO :self._jump_to,
            LowLevelILOperation.LLIL_CALL : self._call,
            LowLevelILOperation.LLIL_RET : self._ret,

            LowLevelILOperation.LLIL_TEST_BIT : self._test_bit,
            LowLevelILOperation.LLIL_SYSCALL : self._syscall,
            LowLevelILOperation.LLIL_BP : self._bp,
            LowLevelILOperation.LLIL_TRAP : self._trap,
            LowLevelILOperation.LLIL_UNDEF : self._undef,
            LowLevelILOperation.LLIL_UNIMPL : self._unimpl,
            LowLevelILOperation.LLIL_UNIMPL_MEM : self._unimpl_mem
        }


    def transfer (self, llil, data=None) :
        print self.optable[llil.operation]
        return self.optable[llil.operation](llil, data)


    def join_lhs_rhs (self, lhs, rhs) :
        pass


    def join (self, data_list) :
        '''
        Data Flow Analysis join function. Keep in mind if there are no predecessors
        this will be an empty list.
        '''
        if len(data_list) < 2 :
            return data_list

        elif len(data_list) > 2 :
            lhs = data_list[0]
            rhs = data_list[1]
            return self.join(self.join_lhs_rhs(lhs, rhs), data_list[2:])

        else :
            return self.join_lhs_rhs(data_list[0], data_list[1])


    def state_equivalence_lhs_rhs (self, lhs, rhs) :
        '''
        Returns True if the two states are equivalent, False otherwise.
        '''
        pass


    def state_equivalence (self, data_list) :
        if len(data_list) < 2 :
            return True

        if len(data_list) > 2 :
            lhs = data_list[0]
            rhs = data_list[1]
            return self.state_equivalence(self.state_equivalence_lhs_rhs(lhs, rhs), \
                                          data_list[2:])

        return self.state_equivalence_lhs_rhs(data_list[0], data_list[1])


    def fixpoint_forward (self, graph) :
        '''
        Dataflow analysis forward until a fixpoint is reached.
        '''
        fixpoint = {}
        queue = []

        for index in graph.vertices :
            fixpoint[index] = None
            queue.append(index)

        while len(queue) > 0 :
            index = queue[0]
            queue = queue[1:]

            vertex = graph.get_vertex_from_index(index)
            pred_fixpoints = map(lambda x: copy.deepcopy(fixpoint[x]), \
                                 vertex.get_predecessor_indices())
            rolling = self.join(pred_fixpoints)

            for ins in vertex.data :
                rolling = self.transfer(ins, rolling)

            if rolling != fixpoint[index] :
                fixpoint[index] = rolling

                queue += vertex.get_successor_indices()

        return fixpoint


    def map_instructions (self, graph) :
        '''
        Returns a dict of instruction addresses/indices and their values from
        this analysis
        '''

        result = {}

        for index in graph.vertices :
            vertex = graph.get_vertex_from_index(index)

            for ins in vertex.data :
                result[ins.address] = self.transfer(ins)

        return result


    def _arith (self, llil, data=None) :
        '''
        This function is executed over LLIL arithmetic instructions, unless a
        handler for a specific arithmetic instruction is given. The arithmetic
        instructions are:

        LLIL_ADD, LLIL_ADC, LLIL_SUB, LLIL_SBB, LLIL_AND, LLIL_OR, LLIL_XOR,
        LLIL_LSL, LLIL_LSR, LLIL_ASR, LLIL_ROL, LLIL_RLC, LLIL_ROR, LLIL_RRC,
        LLIL_MUL, LLIL_DIVU, LLIL_DIVS, LLIL_MODU, LLIL_MODS
        '''
        log.log_error("_arith unsupported for this analysis")

    def _add (self, llil, data=None) :
        return self._arith(llil)

    def _adc (self, llil, data=None) :
        return self._arith(llil)

    def _sub (self, llil, data=None) :
        return self._arith(llil)

    def _sbb (self, llil, data=None) :
        return self._arith(llil)

    def _and (self, llil, data=None) :
        return self._arith(llil)

    def _or (self, llil, data=None) :
        return self._arith(llil)

    def _xor (self, llil, data=None) :
        return self._arith(llil)

    def _lsl (self, llil, data=None) :
        return self._arith(llil)

    def _lsr (self, llil, data=None) :
        return self._arith(llil)

    def _asr (self, llil, data=None) :
        return self._arith(llil)

    def _rol (self, llil, data=None) :
        return self._arith(llil)

    def _rlc (self, llil, data=None) :
        return self._arith(llil)

    def _ror (self, llil, data=None) :
        return self._arith(llil)

    def _rrc (self, llil, data=None) :
        return self._arith(llil)

    def _mul (self, llil, data=None) :
        return self._arith(llil)

    def _divu (self, llil, data=None) :
        return self._arith(llil)

    def _divs (self, llil, data=None) :
        return self._arith(llil)

    def _modu (self, llil, data=None) :
        return self._arith(llil)

    def _mods (self, llil, data=None) :
        return self._arith(llil)

    def _arith_dp (self, llil, data=None) :
        '''
        This function is executed for LLIL_DP arithmetic instructions, unless a
        handler for a specific arithmetic instruction is given. The instructions
        handled here are:

        LLIL_MULU_DP, LLIL_MULS_DP, LLIL_DIVU_DP, LLIL_DIVS_DP, LLIL_MODU_DP,
        LLIL_MODS_DP
        '''
        log.log_error("_arith_dp unsupported for this analysis")

    def _mulu_dp (self, llil, data=None) :
        return self._arith_dp(llil)

    def _muls_dp (self, llil, data=None) :
        return self._arith_dp(llil)

    def _divu_dp (self, llil, data=None) :
        return self._arith_dp(llil)

    def _divs_dp (self, llil, data=None) :
        return self._arith_dp(llil)

    def _modu_dp (self, llil, data=None) :
        return self._arith_dp(llil)

    def _mods_dp (self, llil, data=None):
        return self._arith_dp(llil)

    def _unary (self, llil, data=None) :
        '''
        This function is executed for LLIL unary arithmetic instructions, unless
        a handler for a specific unary arithmetic instruction is given. The
        instructions handled here are:

        LLIL_NEG, LLIL_NOT, LLIL_SX, LLIL_ZX, LLIL_SET_REG, LLIL_SET_REG_SPLIT
        '''
        log.log_error("_unary unsupported for this analysis")

    def _neg (self, llil, data=None) :
        return self._unary(llil)

    def _not (self, llil, data=None) :
        return self._unary(llil)

    def _sx (self, llil, data=None) :
        return self._unary(llil)

    def _zx (self, llil, data=None) :
        return self._unary(llil)

    def _set_reg (self, llil, data=None) :
        return self._unary(llil)

    def _set_reg_split (self, llil, data=None) :
        return self._unary(llil)

    def _cmp (self, llil, data=None) :
        '''
        This function is executed for LLIL comparison instructions, unless a
        handler for a specific comparison instruction is given. The instructions
        handled are:

        LLIL_CMP_E, LLIL_CMP_NE, LLIL_CMP_SLT, LLIL_CMP_ULT, LLIL_CMP_SLE,
        LLIL_CMP_ULE, LLIL_CMP_SGE, LLIL_CMP_UGE, LLIL_CMP_SGT, LLIL_CMP_UGT
        '''
        log.log_error("_cmp not supported for this analysis")

    def _cmp_e (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_ne (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_slt (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_ult (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_sle (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_ule (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_sge (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_uge (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_sgt (self, llil, data=None) :
        return self._cmp(llil)

    def _cmp_ugt (self, llil, data=None) :
        return self._cmp(llil)

    '''
    These ops should hold values
    '''

    def _reg (self, llil, data=None) :
        log.log_error("_reg not supported for this analysis")

    def _const (self, llil, data=None) :
        log.log_error("_const not supported for this analysis")

    def _flag_cond (self, llil, data=None) :
        log.log_error("_flag_cond not supported for this analysis")

    def _flag (self, llil, data=None) :
        log.log_error("_flag not supported for this analysis")

    '''
    Memory ops go here
    '''

    def _load (self, llil, data=None) :
        log.log_error("_load not supported for this analysis")

    def _store (self, llil, data=None) :
        log.log_error("_store not supported for this analysis")

    def _push (self, llil, data=None) :
        log.log_error("_push not supported for this analysis")

    def _pop (self, llil, data=None) :
        log.log_error("_pop not supported for this analysis")

    '''
    Control-Flow ps go here
    '''

    def _noret (self, llil, data=None) :
        log.log_error("_noret not supported for this analysis")

    def _goto (self, llil, data=None) :
        log.log_error("_gogo not supported for this analysis")

    def _If (self, llil, data=None) :
        log.log_error("_If not supported for this analysis")

    def _bool_to_int (self, llil, data=None) :
        log.log_error("_bool_to_int not supported for this analysis")

    def _jump (self, llil, data=None) :
        log.log_error("_jump not supported for this analysis")

    def _jump_to (self, llil, data=None) :
        log.log_error("_jump_to not supported for this analysis")

    def _call (self, llil, data=None) :
        log.log_error("_call not supported for this analysis")

    def _ret (self, llil, data=None) :
        log.log_error("_ret not supported for this analysis")

    '''
    Miscellaneous ops
    '''

    def _test_bit (self, llil, data=None) :
        log.log_error("_test_bit not supported for this analysis")

    def _syscall (self, llil, data=None) :
        log.log_error("_syscall not supported for this analysis")

    def _bp (self, llil, data=None) :
        log.log_error("_bp not supported for this analysis")

    def _trap (self, llil, data=None) :
        log.log_error("_trap not supported for this analysis")

    def _undef (self, llil, data=None) :
        log.log_error("_undef not supported for this analysis")

    def _unimpl (self, llil, data=None) :
        log.log_error("_unimpl not supported for this analysis")

    def _unimpl_mem (self, llil, data=None) :
        log.log_error("_unimpl_mem not supported for this analysis")
