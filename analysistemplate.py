import analysis


class AnalysisTemplate (analysis.AnalysisModel) :
    def __init__ (self) :
        super(ReachingDefinitions, self).__init__()
        self.llil_handler_print = True

    def join_lhs_rhs (self, lhs, rhs) :
        pass

    def state_equivalence_lhs_rhs (self, lhs, rhs) :
        pass

    def _arith (self, llil, data=None) :
        if self.llil_handler_print :
            print '_arith', llil

    def _arith_db (self, llil, data=None) :
        if self.llil_handler_print :
            print '_arith_dp', llil

    def _unary (self, llil, data=None) :
        if self.llil_handler_print :
            print '_unary', llil

    def _cmp (self, llil, data=None) :
        if self.llil_handler_print :
            print '_cmp', llil

    def _reg (self, llil, data=None) :
        if self.llil_handler_print :
            print '_reg', llil

    def _const (self, llil, data=None) :
        if self.llil_handler_print :
            print '_const', llil

    def _flag_cond (self, llil, data=None) :
        if self.llil_handler_print :
            print '_flag_cond', llil

    def _flag (self, llil, data=None) :
        if self.llil_handler_print :
            print '_flag', llil

    def _load (self, llil, data=None) :
        if self.llil_handler_print :
            print '_load', llil

    def _store (self, llil, data=None) :
        if self.llil_handler_print :
            print '_store', llil

    def _push (self, llil, data=None) :
        if self.llil_handler_print :
            print '_push', llil

    def _pop (self, llil, data=None) :
        if self.llil_handler_print :
            print '_pop', llil

    def _noret (self, llil, data=None) :
        if self.llil_handler_print :
            print '_noret', llil

    def _goto (self, llil, data=None) :
        if self.llil_handler_print :
            print '_goto', llil

    def _If (self, llil, data=None) :
        if self.llil_handler_print :
            print '_If', llil

    def _bool_to_int (self, llil, data=None) :
        if self.llil_handler_print :
            print '_bool_to_int', llil

    def _jump (self, llil, data=None) :
        if self.llil_handler_print :
            print '_jump', llil

    def _jump_to (self, llil, data=None) :
        if self.llil_handler_print :
            print '_jump_to', llil

    def _call (self, llil, data=None) :
        if self.llil_handler_print :
            print '_call', llil

    def _ret (self, llil, data=None) :
        if self.llil_handler_print :
            print '_ret', llil

    def _test_bit (self, llil, data=None) :
        if self.llil_handler_print :
            print '_test_bit', llil

    def _syscall (self, llil, data=None) :
        if self.llil_handler_print :
            print '_syscall', llil

    def _bp (self, llil, data=None) :
        if self.llil_handler_print :
            print '_bp', llil

    def _trap (self, llil, data=None) :
        if self.llil_handler_print :
            print '_trap', llil

    def _undef (self, llil, data=None) :
        if self.llil_handler_print :
            print '_undef', llil

    def _unimpl (self, llil, data=None) :
        if self.llil_handler_print :
            print '_unimpl'

    def _unimpl_mem (self, llil, data=None) :
        if self.llil_handler_print :
            print '_unimpl_mem'
