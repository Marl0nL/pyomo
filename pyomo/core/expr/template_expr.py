# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import dis
import itertools
import logging
import sys
import builtins
from contextlib import contextmanager, nullcontext

from pyomo.common.collections import MutableMapping
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.gc_manager import PauseGC
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
    ARG_TYPE,
    NumericExpression,
    Numeric_NPV_Mixin,
    SumExpression,
    mutable_expression,
    register_arg_type,
    _balanced_parens,
)
from pyomo.core.expr.numvalue import (
    NumericValue,
    native_types,
    nonpyomo_leaf_types,
    as_numeric,
    value,
    is_constant,
)
from pyomo.core.expr.relational_expr import (
    tuple_to_relational_expr,
    EqualityExpression,
    NotEqualExpression,
    InequalityExpression,
    RangedExpression,
    set_index_predicate_bool_hook,
)
from pyomo.core.expr.visitor import (
    ExpressionReplacementVisitor,
    StreamBasedExpressionVisitor,
    expression_to_string,
    _ToStringVisitor,
)

# Deferred imports to break circular dependencies
pyomo_core_base_set, _ = attempt_import('pyomo.core.base.set')
pyomo_core_base_param, _ = attempt_import('pyomo.core.base.param')

logger = logging.getLogger(__name__)

# When True, ``templatize_rule`` does not log an ERROR for a rule that fails to
# templatize.  Opportunistic callers (the ``TEMPLATIZE_CONSTRAINTS`` construction
# fast path and the vectorized-compile path in ``pyomo.contrib.vector``) set this
# via :func:`suppress_templatization_errors`, because for them a rule that does
# not templatize is an expected, silently-handled fallback -- not an error.
_SUPPRESS_TEMPLATIZATION_ERRORS = False


@contextmanager
def suppress_templatization_errors():
    """Silence the ERROR ``templatize_rule`` logs when a rule does not templatize.

    Use around an *opportunistic* templatization attempt whose failure is handled
    by falling back to classic construction, so the fallback stays quiet.
    """
    global _SUPPRESS_TEMPLATIZATION_ERRORS
    prev = _SUPPRESS_TEMPLATIZATION_ERRORS
    _SUPPRESS_TEMPLATIZATION_ERRORS = True
    try:
        yield
    finally:
        _SUPPRESS_TEMPLATIZATION_ERRORS = prev


# --------------------------------------------------------------------------- #
# Index-predicate capture (Phase 3b: filtered sums + index conditionals)
# --------------------------------------------------------------------------- #
#
# By default a predicate on index values (``if j != n``, ``if t > 0``) raises
# when it reaches a boolean context during templatization, because the index
# values are not yet known -- so a rule containing a filtered sum or an index
# conditional does not templatize and falls back to classic per-index
# construction.  Under the opt-in vectorized-construction switch we instead
# *capture* such predicates: a filter inside ``sum(... for j in J if PRED)`` is
# recorded and attached to the ``TemplateSumExpression`` (so the extractor can
# evaluate it as a NumPy mask over the sum grid), and an index conditional is
# resolved by the ``conditional_policy`` (Phase-3b conditional replay).
#
# This is entirely gated on ``_INDEX_PREDICATE_CAPTURE`` being non-None, which
# only happens for the duration of a vectorized templatization: with the switch
# off, ``__bool__`` behaves exactly as in stock Pyomo.
_INDEX_PREDICATE_CAPTURE = None


class _IndexPredicateCapture:
    """State for capturing index-value predicates during one templatization.

    ``filters`` is a stack that :meth:`_template_iter_context.sum_template`
    snapshots and drains around each ``next(generator)`` to collect the filter
    predicate(s) of the sum it is building (``if j != n`` -> a conjunction of the
    recorded comparisons).  A predicate encountered *outside* a sum filter is an
    index conditional selector, handled by ``conditional_policy`` (a callable
    ``(expr, capture) -> bool | None``; ``None`` declines, so the family falls
    back).  ``unsupported`` latches when a conditional is seen with no policy.
    """

    __slots__ = ('filters', 'in_filter', 'conditional_policy', 'unsupported')

    def __init__(self, conditional_policy=None):
        self.filters = []
        self.in_filter = 0
        self.conditional_policy = conditional_policy
        self.unsupported = False


_COMPARISON_OPCODES = frozenset({'COMPARE_OP', 'CONTAINS_OP', 'IS_OP'})


def _count_comparisons(code):
    """Number of comparison opcodes in a (generator) code object."""
    try:
        return sum(
            1 for i in dis.get_instructions(code) if i.opname in _COMPARISON_OPCODES
        )
    except TypeError:
        return -1


def _filter_is_pure_conjunction(generator, n_captured):
    """True if the generator's filter is a pure conjunction of comparisons.

    We capture a filter by making each comparison's ``__bool__`` return True, but
    Python's ``or`` / ``not`` short-circuit, so a disjunction (``k < n or k > n``)
    would silently capture only its first clause -- a wrong (too-permissive)
    filter.  A conjunction (``a and b``, or chained ``if a if b``) instead
    evaluates *every* comparison, so the count of comparison opcodes in the
    generator equals the number of predicates we captured.  If they differ, the
    filter short-circuited (an ``or`` / mixed boolean) and is not vectorizable as
    a conjunction -- the caller falls back to classic construction.
    """
    code = getattr(generator, 'gi_code', None)
    if code is None:
        return False
    ncomp = _count_comparisons(code)
    return ncomp == n_captured


def _index_predicate_hook(expr):
    """``RelationalExpression.__bool__`` hook: capture an index predicate.

    Returns the boolean the calling context should observe (after recording the
    predicate), or ``None`` to decline -- in which case the normal "cannot
    convert to bool" error is raised and the family falls back to classic
    construction.  Declines for any predicate that touches a variable (only pure
    index-value predicates are vectorizable).
    """
    cap = _INDEX_PREDICATE_CAPTURE
    if cap is None:
        return None
    if expr.is_potentially_variable():
        return None
    if cap.in_filter:
        # A generator filter inside a sum: record it and let the summand be
        # yielded once (the extractor applies the mask over the sum grid).
        cap.filters.append(expr)
        return True
    policy = cap.conditional_policy
    if policy is None:
        cap.unsupported = True
        return None
    return policy(expr, cap)


@contextmanager
def capture_index_predicates(conditional_policy=None):
    """Enable index-predicate capture for the enclosed templatization.

    Installs the :func:`_index_predicate_hook` on relational ``__bool__`` and a
    fresh :class:`_IndexPredicateCapture`.  Restores the prior state on exit, so
    it nests and never leaks into stock behaviour.
    """
    global _INDEX_PREDICATE_CAPTURE
    prev = _INDEX_PREDICATE_CAPTURE
    prev_hook = set_index_predicate_bool_hook(_index_predicate_hook)
    _INDEX_PREDICATE_CAPTURE = _IndexPredicateCapture(conditional_policy)
    try:
        yield _INDEX_PREDICATE_CAPTURE
    finally:
        _INDEX_PREDICATE_CAPTURE = prev
        set_index_predicate_bool_hook(prev_hook)


def _validate_generator(generator):
    # We are worried about users writing things like
    #
    #    sum(m.x[i, j] for i in [1, 2, 3] for j in m.J)
    # or
    #    sum(m.x[j] for i in [1, 2, 3] for j in m.J[i])
    #
    # If they do, we will not "see" the "i" as an IndexTemplate, so the
    # expression would be reduced to
    #
    #    sum(m.x[1, j] for j in m.J)
    # and
    #    sum(m.x[j] for j in m.J[1])
    #
    # To guard against this, we will look into the generator code, and
    # if there are any local variables declared that are not
    # IndexTemplate objects (or tuples of them), then we will throw up
    # our hands and expand the sum:
    for lvar_name in generator.gi_frame.f_code.co_varnames:
        if lvar_name == '.0':
            # Skip the outer generator object
            continue
        lvar = generator.gi_frame.f_locals.get(lvar_name, None)
        if lvar.__class__ is IndexTemplate:
            continue
        if lvar.__class__ is tuple and all(i.__class__ is IndexTemplate for i in lvar):
            continue
        return False
    return True


def _validate_map(generator):
    # We would love to validate that the map is actually iterating over
    # Pyomo Set objects (and yielding IndexTemplate objects), but there
    # doesn't appear to be a way to interrogate the results of the list
    # / generator that the map is iterating over.  So, we will just have
    # to trust the user <shudder>.
    #
    # FIXME: rework IndexedComponent to return custom generators that
    # wrap map so we can only accept them and not all maps?
    return True


# it is not clear what to import to get to the built-in "generator"
# type.  We will just create a generator and query its __class__
generator_validators = {
    (_ for _ in ()).__class__: _validate_generator,
    map: _validate_map,
}


class _NotSpecified:
    pass


class GetItemExpression(ExpressionBase):
    """
    Expression to call :func:`__getitem__` on the base object.
    """

    __slots__ = ()
    PRECEDENCE = 1

    def __new__(cls, args=()):
        if cls is not GetItemExpression:
            return super().__new__(cls)
        npv_args = not any(
            hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable()
            for arg in args
        )
        try:
            component = _reduce_template_to_component(args[0])
            cdata = component._ComponentDataClass(component)
            if cdata.is_numeric_type():
                if npv_args and not cdata.is_potentially_variable():
                    return super().__new__(NPV_Numeric_GetItemExpression)
                else:
                    return super().__new__(Numeric_GetItemExpression)
            if cdata.is_logical_type():
                if npv_args and not cdata.is_potentially_variable():
                    return super().__new__(NPV_Boolean_GetItemExpression)
                else:
                    return super().__new__(Boolean_GetItemExpression)
        except (AttributeError, TypeError):
            # TypeError: error reducing to a component (usually due to
            #     unbounded domain on a Var used in a GetItemExpression)
            # AttributeError: resolved component did not support the
            #     PyomoObject API
            pass
        if npv_args:
            return super().__new__(NPV_Structural_GetItemExpression)
        else:
            return super().__new__(Structural_GetItemExpression)

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError()
        return GetAttrExpression((self, attr))

    def __iter__(self):
        return iter(value(self))

    def __len__(self):
        return len(value(self))

    def getname(self, *args, **kwds):
        return self._args_[0].getname(*args, **kwds)

    def nargs(self):
        return len(self._args_)

    def _is_fixed(self, values):
        if not all(values[1:]):
            return False
        _true = lambda: True
        return all(getattr(x, 'is_fixed', _true)() for x in values[0].values())

    def _to_string(self, values, verbose, smap):
        values = tuple(_[1:-1] if _[0] == '(' and _[-1] == ')' else _ for _ in values)
        if verbose:
            return "getitem(%s, %s)" % (values[0], ', '.join(values[1:]))
        return "%s[%s]" % (values[0], ','.join(values[1:]))

    def _resolve_template(self, args):
        return args[0].__getitem__(args[1:])

    def _apply_operation(self, result):
        return result[0].__getitem__(result[1:])


class Numeric_GetItemExpression(GetItemExpression, NumericExpression):
    __slots__ = ()

    def nargs(self):
        return len(self._args_)

    def _compute_polynomial_degree(self, result):
        if any(x != 0 for x in result[1:]):
            return None
        ans = 0
        for x in result[0].values():
            if x.__class__ in nonpyomo_leaf_types or not hasattr(
                x, 'polynomial_degree'
            ):
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans


class NPV_Numeric_GetItemExpression(Numeric_NPV_Mixin, Numeric_GetItemExpression):
    __slots__ = ()


class Boolean_GetItemExpression(GetItemExpression, BooleanExpression):
    __slots__ = ()


class NPV_Boolean_GetItemExpression(NPV_Mixin, Boolean_GetItemExpression):
    __slots__ = ()


class Structural_GetItemExpression(ExpressionArgs_Mixin, GetItemExpression):
    __slots__ = ()


class NPV_Structural_GetItemExpression(NPV_Mixin, Structural_GetItemExpression):
    __slots__ = ()


class GetAttrExpression(ExpressionBase):
    """
    Expression to call :func:`__getattr__` on the base object.
    """

    __slots__ = ()
    PRECEDENCE = 1

    def __new__(cls, args=()):
        if cls is not GetAttrExpression:
            return super().__new__(cls)
        # Ironically, we need to actually create this object in order to
        # determine what the class for this object should be.
        if args[0].is_potentially_variable():
            self = Structural_GetAttrExpression(args)
        else:
            self = NPV_Structural_GetAttrExpression(args)
        try:
            attr = _reduce_template_to_component(self)
            if attr.is_numeric_type():
                if attr.is_potentially_variable() or self.is_potentially_variable():
                    return super().__new__(Numeric_GetAttrExpression)
                else:
                    return super().__new__(NPV_Numeric_GetAttrExpression)
            elif attr.is_logical_type():
                if attr.is_potentially_variable() or self.is_potentially_variable():
                    return super().__new__(Boolean_GetAttrExpression)
                else:
                    return super().__new__(NPV_Boolean_GetAttrExpression)
        except (AttributeError, TypeError):
            # TypeError: error reducing to a component (usually due to
            #     unbounded domain on a Var used in a GetItemExpression)
            # AttributeError: resolved component did not support the
            #     PyomoObject API
            pass
        return self

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError()
        return GetAttrExpression((self, attr))

    def __getitem__(self, *idx):
        return GetItemExpression((self,) + idx)

    def __iter__(self):
        return iter(value(self))

    def __len__(self):
        return len(value(self))

    def __call__(self, *args, **kwargs):
        """
        Return the value of this object.
        """
        # Backwards compatibility with __call__(exception):
        #
        # TODO: deprecate (then remove) evaluating expressions by
        # "calling" them.
        #
        # [ESJ 3/25/25]: Note that since this always calls the ExpressionBase
        # implementation of __call__ if 'exception' is specified, we need not
        # check the type of the exception arg here--it will get checked in the
        # base class.
        try:
            if not args:
                if not kwargs:
                    return super().__call__()
                elif len(kwargs) == 1 and 'exception' in kwargs:
                    return super().__call__(**kwargs)
            elif (
                not kwargs and len(args) == 1 and (args[0] is True or args[0] is False)
            ):
                return super().__call__(*args)
        except TemplateExpressionError:
            pass
        # Note: the only time we will implicitly create a CallExpression
        # node is directly after a GetAttrExpression: that is, someone
        # got the attribute (method) and is now calling it.
        # Implementing the auto-generation of CallExpression in other
        # contexts is likely to be confounded with evaluating expressions.
        return CallExpression((self,) + args, kwargs)

    def getname(self, *args, **kwds):
        return 'getattr'

    def nargs(self):
        return 2

    def _apply_operation(self, result):
        obj, attr = result
        return getattr(obj, attr)

    def _to_string(self, values, verbose, smap):
        assert len(values) == 2
        if verbose:
            return "getattr(%s, %s)" % tuple(values)
        # Note that the string argument for getattr comes quoted, so we
        # need to remove the quotes.
        attr = values[1]
        if attr[0] in '\"\'' and attr[0] == attr[-1]:
            attr = attr[1:-1]
        return "%s.%s" % (values[0], attr)

    def _resolve_template(self, args):
        return getattr(*args)


class Numeric_GetAttrExpression(GetAttrExpression, NumericExpression):
    __slots__ = ()

    def _compute_polynomial_degree(self, result):
        if result[1] != 0:
            return None
        return result[0]


class NPV_Numeric_GetAttrExpression(Numeric_NPV_Mixin, Numeric_GetAttrExpression):
    __slots__ = ()


class Boolean_GetAttrExpression(GetAttrExpression, BooleanExpression):
    __slots__ = ()


class NPV_Boolean_GetAttrExpression(NPV_Mixin, Boolean_GetAttrExpression):
    __slots__ = ()


class Structural_GetAttrExpression(ExpressionArgs_Mixin, GetAttrExpression):
    __slots__ = ()


class NPV_Structural_GetAttrExpression(NPV_Mixin, Structural_GetAttrExpression):
    __slots__ = ()


class CallExpression(NumericExpression):
    """
    Expression to call :func:`__call__` on the base object.
    """

    __slots__ = ('_kwds',)
    PRECEDENCE = None

    def __init__(self, args, kwargs):
        self._args_ = tuple(args) + tuple(kwargs.values())
        self._kwds = tuple(kwargs.keys())

    def nargs(self):
        return len(self._args_)

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError()
        return GetAttrExpression((self, attr))

    def __getitem__(self, *idx):
        return GetItemExpression((self,) + idx)

    def __iter__(self):
        return iter(value(self))

    def __len__(self):
        return len(value(self))

    def getname(self, *args, **kwds):
        return 'call'

    def _compute_polynomial_degree(self, result):
        return None

    def _apply_operation(self, result):
        na = len(self._args_) - len(self._kwds)
        return result[0](*result[1:na], **dict(zip(self._kwds, result[na:])))

    def _to_string(self, values, verbose, smap):
        na = len(self._args_) - len(self._kwds)
        args = ', '.join(values[1:na])
        if self._kwds:
            if na > 1:
                args += ', '
            args += ', '.join(
                f'{key}={val}' for key, val in zip(self._kwds, values[na:])
            )
        if verbose:
            return f"call({values[0]}, {args})"
        return f"{values[0]}({args})"

    def _resolve_template(self, args):
        return self._apply_operation(args)


class _TemplateSumExpression_argList:
    """A virtual list to represent the expanded SumExpression args

    This class implements a "virtual args list" for
    TemplateSumExpressions without actually generating the expanded
    expression.  It can be accessed either in "one-pass" without
    generating a list of template argument values (more efficient), or
    as a random-access list (where it will have to create the full list
    of argument values (less efficient).

    The instance can be used as a context manager to both lock the
    IndexTemplate values within this context and to restore their original
    values upon exit.

    It is (intentionally) not iterable.

    """

    def __init__(self, TSE):
        self._tse = TSE
        self._i = 0
        self._init_vals = None
        self._iter = self._get_iter()
        self._lock = None

    def __len__(self):
        return self._tse.nargs()

    def __getitem__(self, i):
        if self._i == i:
            self._set_iter_vals(next(self._iter))
            self._i += 1
        elif self._i is not None:
            # Switch to random-access mode.  If we have already
            # retrieved one of the indices, then we need to regenerate
            # the iterator from scratch.
            self._iter = list(self._get_iter() if self._i else self._iter)
            self._set_iter_vals(self._iter[i])
        else:
            self._set_iter_vals(self._iter[i])
        return self._tse._local_args_[0]

    def __enter__(self):
        self._lock = self
        self._lock_iters()

    def __exit__(self, exc_type, exc_value, tb):
        self._unlock_iters()
        self._lock = None

    def _get_iter(self):
        # Note: by definition, all _set pointers within an itergroup
        # point to the same Set
        _sets = tuple(iterGroup[0]._set for iterGroup in self._tse._iters)
        return itertools.product(*_sets)

    def _lock_iters(self):
        self._init_vals = tuple(
            tuple(it.lock(self._lock) for it in iterGroup)
            for iterGroup in self._tse._iters
        )

    def _unlock_iters(self):
        self._set_iter_vals(self._init_vals)
        for iterGroup in self._tse._iters:
            for it in iterGroup:
                it.unlock(self._lock)

    def _set_iter_vals(self, val):
        for i, iterGroup in enumerate(self._tse._iters):
            if len(iterGroup) == 1:
                iterGroup[0].set_value(val[i], self._lock)
            else:
                for j, v in enumerate(val[i]):
                    iterGroup[j].set_value(v, self._lock)


class TemplateSumExpression(NumericExpression):
    """
    Expression to represent an unexpanded sum over one or more sets.
    """

    __slots__ = ('_iters', '_local_args_', '_filter')
    PRECEDENCE = 1

    def __init__(self, args, _iters, _filter=None):
        assert len(args) == 1
        self._args_ = args
        self._iters = _iters
        # Phase 3b: an optional filter for ``sum(... for j in J if PRED)`` -- a
        # tuple of index-value predicate expressions, interpreted as a
        # conjunction.  ``None`` means an unfiltered sum (stock behaviour).
        self._filter = _filter

    def nargs(self):
        # Note: by definition, all _set pointers within an itergroup
        # point to the same Set
        ans = 1
        for iterGroup in self._iters:
            ans *= len(iterGroup[0]._set)
        return ans

    @property
    def args(self):
        return _TemplateSumExpression_argList(self)

    @property
    def _args_(self):
        return _TemplateSumExpression_argList(self)

    @_args_.setter
    def _args_(self, args):
        self._local_args_ = args

    def template_args(self):
        ans = list(self._local_args_)
        for itergroup in self._iters:
            ans.append(itergroup[0]._set)
        return tuple(ans)

    def template_iters(self):
        return self._iters

    def template_filter(self):
        """The captured filter predicates (a conjunction), or ``None``."""
        return self._filter

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._iters, self._filter)

    def getname(self, *args, **kwds):
        return "SUM"

    def is_potentially_variable(self):
        if any(
            arg.is_potentially_variable()
            for arg in self._local_args_
            if arg.__class__ not in nonpyomo_leaf_types
        ):
            return True
        return False

    def _is_fixed(self, values):
        return all(values)

    def _compute_polynomial_degree(self, result):
        if None in result:
            return None
        return result[0]

    def _apply_operation(self, result):
        return sum(result)

    def to_string(self, verbose=None, smap=None):
        ans = ''
        assert len(self._local_args_) == 1
        val = expression_to_string(self._local_args_[0], verbose=verbose, smap=smap)
        if val[0] == '(' and val[-1] == ')' and _balanced_parens(val[1:-1]):
            val = val[1:-1]
        iterStrGenerator = (
            (
                ', '.join(
                    (smap.getSymbol(i) if smap is not None else str(i))
                    for i in iterGroup
                ),
                (
                    iterGroup[0]._set.to_string(verbose=verbose, smap=smap)
                    if hasattr(iterGroup[0]._set, 'to_string')
                    else (
                        smap.getSymbol(iterGroup[0]._set)
                        if smap is not None
                        else str(iterGroup[0]._set)
                    )
                ),
            )
            for iterGroup in self._iters
        )
        filtStr = ''
        if self._filter:
            preds = ' and '.join(
                expression_to_string(p, verbose=verbose, smap=smap)
                for p in self._filter
            )
            filtStr = ' if ' + preds
        if verbose:
            iterStr = ', '.join('iter(%s, %s)' % x for x in iterStrGenerator)
            return 'templatesum(%s, %s%s)' % (val, iterStr, filtStr)
        else:
            iterStr = ' '.join('for %s in %s' % x for x in iterStrGenerator)
            return 'SUM(%s %s%s)' % (val, iterStr, filtStr)

    def _resolve_template(self, args):
        with mutable_expression() as e:
            for arg in args:
                e += arg
        if e.nargs() > 1:
            return e
        elif not e.nargs():
            return 0
        else:
            return e.arg(0)


# FIXME: This is a hack to get certain complex cases to print without error
_ToStringVisitor._leaf_node_types.add(TemplateSumExpression)


class IndexTemplate(NumericValue):
    """A "placeholder" for an index value in template expressions.

    This class is a placeholder for an index value within a template
    expression.  That is, given the expression template for "m.x[i]",
    where `m.z` is indexed by `m.I`, the expression tree becomes:

    _GetItem:
       - m.x
       - IndexTemplate(_set=m.I, _value=None)

    Constructor Arguments:
       _set: the Set from which this IndexTemplate can take values
    """

    __slots__ = ('_set', '_value', '_index', '_id', '_group', '_lock')

    def __init__(self, _set, index=0, _id=None, _group=None):
        self._set = _set
        self._value = _NotSpecified
        self._index = index
        self._id = _id
        self._group = _group
        self._lock = None

    def __deepcopy__(self, memo):
        # Because we leverage deepcopy for expression/component cloning,
        # we need to see if this is a Component.clone() operation and
        # *not* copy the template.
        #
        # TODO: JDS: We should consider converting the IndexTemplate to
        # a proper Component: that way it could leverage the normal
        # logic of using the parent_block scope to dictate the behavior
        # of deepcopy.
        if '__block_scope__' in memo:
            memo[id(self)] = self
            return self
        #
        # "Normal" deepcopying outside the context of pyomo.
        #
        return super().__deepcopy__(memo)

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is _NotSpecified:
            if exception:
                raise TemplateExpressionError(
                    self, "Evaluating uninitialized IndexTemplate (%s)" % (self,)
                )
            return None
        else:
            return self._value

    def _resolve_template(self, args):
        assert not args
        return self()

    def is_fixed(self):
        """
        Returns True because this value is fixed.
        """
        return True

    def is_potentially_variable(self):
        """Returns False because index values cannot be variables.

        The IndexTemplate represents a placeholder for an index value
        for an IndexedComponent, and at the moment, Pyomo does not
        support variable indirection.
        """
        return False

    def __ne__(self, other):
        # Phase 3b: under the vectorized-construction switch, ``j != n`` in a
        # generator filter must build a NotEqualExpression so the capture hook
        # sees the ``!=`` predicate with the right polarity.  Python's default
        # ``__ne__`` is ``not (self == other)``, which would evaluate
        # ``bool(EqualityExpression)`` (an ``==`` node) and lose the negation.
        # Outside capture, we reproduce that stock behaviour exactly.
        if _INDEX_PREDICATE_CAPTURE is not None:
            return NotEqualExpression((self, other))
        return not (self == other)

    def __str__(self):
        return self.getname()

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        if self._id is not None:
            return "_%s" % (self._id,)

        _set_name = self._set.getname(fully_qualified, name_buffer, relative_to)
        if self._index is not None and self._set.dimen != 1:
            _set_name += "(%s)" % (self._index,)
        return "{" + _set_name + "}"

    def set_value(self, values=_NotSpecified, lock=None):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimension
        # 1.  So, for the time being, we will just "trust" the user.
        # After all, the actual Set will raise exceptions if the value
        # is not present.
        if lock is not self._lock:
            raise RuntimeError(
                "The IndexTemplate %s is currently locked by %s and "
                "cannot be set through lock %s" % (self, self._lock, lock)
            )
        if values is _NotSpecified:
            self._value = _NotSpecified
            return
        if type(values) is not tuple:
            values = (values,)
        if self._index is not None:
            if len(values) == 1:
                self._value = values[0]
            else:
                self._value = values[self._index]
        else:
            self._value = values

    def lock(self, lock):
        assert self._lock is None
        self._lock = lock
        return self._value

    def unlock(self, lock):
        assert self._lock is lock
        self._lock = None


# Instead of special-casing _categorize_arg_type for this class, we
# will directly register that it should be treated as an NPV arg
register_arg_type(IndexTemplate, ARG_TYPE.NPV)


class _TemplateResolver(StreamBasedExpressionVisitor):
    def beforeChild(self, node, child, child_idx):
        # Efficiency: do not descend into leaf nodes.
        if type(child) in native_types:
            return False, child
        elif not child.is_expression_type():
            if hasattr(child, '_resolve_template'):
                return False, child._resolve_template(())
            return False, child
        else:
            return True, None

    def exitNode(self, node, args):
        if hasattr(node, '_resolve_template'):
            return node._resolve_template(args)
        if len(args) == node.nargs() and all(a is b for a, b in zip(node.args, args)):
            return node
        if all(map(is_constant, args)):
            return node._apply_operation(args)
        else:
            return node.create_node_with_local_data(args)

    def initializeWalker(self, expr):
        return self.beforeChild(None, expr, None)


def resolve_template(expr):
    """Resolve a template into a concrete expression

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    if resolve_template.visitor is None:
        resolve_template.visitor = _TemplateResolver()
    return resolve_template.visitor.walk_expression(expr)


resolve_template.visitor = None


class _wildcard_info:
    __slots__ = ('iter', 'source', 'value', 'original_value', 'objects')

    def __init__(self, src, obj):
        self.source = src
        self.original_value = obj._value
        self.objects = [obj]
        self.reset()
        if self.original_value in (None, _NotSpecified):
            self.advance()

    def advance(self):
        with _TemplateIterManager.pause():
            self.value = next(self.iter)
        for obj in self.objects:
            obj.set_value(self.value)

    def reset(self):
        # Because we want to actually iterate over the underlying
        # template expression, we will temporarily pause our overrides
        # of sum() and the set iters
        with _TemplateIterManager.pause():
            self.iter = iter(self.source)

    def restore(self):
        for obj in self.objects:
            obj.set_value(self.original_value)


def _reduce_template_to_component(expr):
    """Resolve a template into a concrete component

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    # wildcards holds lists of
    #   [iterator, source, value, orig_value, object0, ...]
    # 'iterator' iterates over 'source' to provide 'value's for each of
    # the 1 or more 'objects'.  Objects can be IndexTemplate objects or
    # (discrete) Variables
    wildcards = []
    wildcard_groups = {}
    level = -1

    def beforeChild(node, child, child_idx):
        # Efficiency: do not descend into leaf nodes.
        if type(child) in native_types:
            return False, child
        elif not child.is_expression_type():
            if hasattr(child, '_resolve_template'):
                try:
                    ans = child._resolve_template(())
                except TemplateExpressionError:
                    # We are attempting "loose" template resolution: for
                    # every unset IndexTemplate, search the underlying
                    # set to find *any* valid match.
                    if child._group not in wildcard_groups:
                        wildcard_groups[child._group] = len(wildcards)
                        info = _wildcard_info(child._set, child)
                        wildcards.append(info)
                    else:
                        info = wildcards[wildcard_groups[child._group]]
                        info.objects.append(child)
                        child.set_value(info.value)
                    ans = child._resolve_template(())
                return False, ans
            if child.is_variable_type():
                if child.domain.isdiscrete():
                    domain = child.domain
                    bounds = child.bounds
                    if bounds != (None, None):
                        try:
                            bounds = pyomo_core_base_set.RangeSet(*bounds, 0)
                            domain = domain & bounds
                        except:
                            pass
                    info = _wildcard_info(domain, child)
                    wildcards.append(info)
                return False, value(child)
            return False, child
        else:
            return True, None

    def exitNode(node, args):
        if hasattr(node, '_resolve_template'):
            return node._resolve_template(args)
        if len(args) == node.nargs() and all(a is b for a, b in zip(node.args, args)):
            return node
        if all(map(is_constant, args)):
            return node._apply_operation(args)
        else:
            return node.create_node_with_local_data(args)

    walker = StreamBasedExpressionVisitor(
        initializeWalker=lambda x: beforeChild(None, x, None),
        beforeChild=beforeChild,
        exitNode=exitNode,
    )
    while 1:
        try:
            with _TemplateIterManager.pause():
                ans = walker.walk_expression(expr)
            break
        except (KeyError, AttributeError):
            # We are attempting "loose" template resolution: for every
            # unset IndexTemplate, search the underlying set to find
            # *any* valid match.
            level = len(wildcards) - 1
            while level >= 0:
                info = wildcards[level]
                try:
                    info.advance()
                    break
                except StopIteration:
                    # Because we want to actually iterate over the
                    # underlying template expression, we will
                    # temporarily pause our overrides of sum() and the
                    # set iters
                    info.reset()
                    info.advance()
                    level -= 1
            if level < 0:
                for info in wildcards:
                    info.restore()
                raise
    for info in wildcards:
        info.restore()
    return ans


class ReplaceTemplateExpression(ExpressionReplacementVisitor):
    template_types = {
        IndexTemplate,
        GetItemExpression,
        Numeric_GetItemExpression,
        NPV_Numeric_GetItemExpression,
        Boolean_GetItemExpression,
        NPV_Boolean_GetItemExpression,
    }

    def __init__(self, substituter, *args, **kwargs):
        kwargs.setdefault('remove_named_expressions', True)
        super().__init__(**kwargs)
        self.substituter = substituter
        self.substituter_args = args

    def beforeChild(self, node, child, child_idx):
        if type(child) in ReplaceTemplateExpression.template_types:
            return False, self.substituter(child, *self.substituter_args)
        return super().beforeChild(node, child, child_idx)


def substitute_template_expression(expr, substituter, *args, **kwargs):
    r"""Substitute IndexTemplates in an expression tree.

    This is a general utility function for walking the expression tree
    and substituting all occurrences of IndexTemplate and
    GetItemExpression nodes.

    Parameters
    ----------
    expr : NumericExpression
        the source template expression

    substituter: Callable
        method taking ``(expression, *args)`` and returning the new object

    \*args:
        positional arguments passed directly to the substituter

    Returns
    -------
    NumericExpression :
        a new expression tree with all substitutions done

    """
    visitor = ReplaceTemplateExpression(substituter, *args, **kwargs)
    return visitor.walk_expression(expr)


class _GetItemIndexer:
    # Note that this class makes the assumption that only one template
    # ever appears in an expression for a single index

    def __init__(self, expr):
        self._base = expr.arg(0)
        self._args = []
        _hash = [id(self._base)]
        for x in expr.args[1:]:
            try:
                logging.disable(logging.CRITICAL)
                val = value(x)
                self._args.append(val)
                _hash.append(val)
            except TemplateExpressionError as e:
                if x is not e.template:
                    raise TypeError(
                        "Cannot use the param substituter with expression "
                        "templates\nwhere the component index has the "
                        "IndexTemplate in an expression.\n\tFound in %s" % (expr,)
                    )
                self._args.append(e.template)
                _hash.append(id(e.template._set))
            finally:
                logging.disable(logging.NOTSET)

        self._hash = tuple(_hash)

    def nargs(self):
        return len(self._args)

    def arg(self, i):
        return self._args[i]

    @property
    def base(self):
        return self._base

    @property
    def args(self):
        return self._args

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if type(other) is _GetItemIndexer:
            return self._hash == other._hash
        else:
            return False

    def __str__(self):
        return "%s[%s]" % (self._base.name, ','.join(str(x) for x in self._args))


def substitute_getitem_with_param(expr, _map):
    """A simple substituter to replace _GetItem nodes with mutable Params.

    This substituter will replace all GetItemExpression nodes with a
    new Param.  For example, this method will create expressions
    suitable for passing to DAE integrators
    """
    if type(expr) is IndexTemplate:
        return expr

    _id = _GetItemIndexer(expr)
    if _id not in _map:
        _map[_id] = pyomo_core_base_param.Param(mutable=True)
        _map[_id].construct()
        _map[_id]._name = "%s[%s]" % (_id.base.name, ','.join(str(x) for x in _id.args))
    return _map[_id]


def substitute_template_with_value(expr):
    """A simple substituter to expand expression for current template

    This substituter will replace all GetItemExpression / IndexTemplate
    nodes with the actual _ComponentData based on the current value of
    the IndexTemplate(s)

    """

    if type(expr) is IndexTemplate:
        return as_numeric(expr())
    else:
        return resolve_template(expr)


class _set_iterator_template_generator:
    """Replacement iterator that returns IndexTemplates

    In order to generate template expressions, we hijack the normal Set
    iteration mechanisms so that this iterator is returned instead of
    the usual iterator.  This iterator will return IndexTemplate
    object(s) instead of the actual Set items the first time next() is
    called.
    """

    def __init__(self, _set, context):
        self._set = _set
        self.context = context

    def __iter__(self):
        return self

    def __next__(self):
        # Prevent context from ever being called more than once
        if self.context is None:
            raise StopIteration()
        context, self.context = self.context, None

        _set = self._set
        if _set.is_expression_type():
            d = _reduce_template_to_component(_set).dimen
        else:
            d = _set.dimen
        grp = context.next_group()
        if type(d) is not int:
            # This covers None (jagged set) and UnknownSetDimen.  In
            # both cases, we will not attempt to unpack the Set and just
            # assume a single index template.
            idx = (IndexTemplate(_set, None, context.next_id(), grp),)
        else:
            idx = tuple(
                IndexTemplate(_set, i, context.next_id(), grp) for i in range(d)
            )
        context.cache.append(idx)
        if len(idx) == 1:
            return idx[0]
        else:
            return idx

    next = __next__


class _template_iter_context:
    """Manage the iteration context when generating templatized rules

    This class manages the context tracking when generating templatized
    rules.  It has two methods (`sum_template` and `get_iter`) that
    replace standard functions / methods (`sum` and
    :py:meth:`_FiniteSetMixin.__iter__`, respectively).  It also tracks
    unique identifiers for IndexTemplate objects and their groupings
    within `sum()` generators.
    """

    def __init__(self):
        self.cache = []
        self._id = 0
        self._group = 0

    def get_iter(self, _set):
        return _set_iterator_template_generator(_set, self)

    def npop_cache(self, n):
        result = self.cache[-n:]
        self.cache[-n:] = []
        return result

    def next_id(self):
        self._id += 1
        return self._id

    def next_group(self):
        self._group += 1
        return self._group

    def sum_template(self, generator):
        try:
            validator = generator_validators[generator.__class__]
        except KeyError:
            # We will only templatize sums over maps and generators.
            # Expand everything else:
            return _TemplateIterManager.builtin_sum(generator)
        # Phase 3b: capture any generator *filter* (``if j != n``) evaluated
        # while advancing the generator, so it can be attached to the
        # TemplateSumExpression.  ``in_filter`` marks the hook that this bool()
        # is a filter (yield the summand, record the predicate) rather than an
        # index conditional; the snapshot/drain isolates *this* sum's filter
        # from any nested sum (whose filter is drained first).
        cap = _INDEX_PREDICATE_CAPTURE
        niters = -len(self.cache)
        if cap is not None:
            filt_mark = len(cap.filters)
            cap.in_filter += 1
            try:
                expr = next(generator)
            finally:
                cap.in_filter -= 1
            _filter = tuple(cap.filters[filt_mark:]) or None
            del cap.filters[filt_mark:]
            if _filter is not None and not _filter_is_pure_conjunction(
                generator, len(_filter)
            ):
                # A disjunction / mixed-boolean filter short-circuited and we did
                # not capture every comparison: reject so the family falls back
                # to classic construction (rather than a wrong, too-loose mask).
                raise TemplateExpressionError(
                    None,
                    "filtered sum with a non-conjunctive predicate "
                    "(or / not / short-circuit) is not vectorizable",
                )
        else:
            _filter = None
            expr = next(generator)
        niters += len(self.cache)
        if niters:
            iters = self.npop_cache(niters)
        else:
            # This didn't generate any new IndexTemplate objects; expand it:
            return _TemplateIterManager.builtin_sum(generator, start=expr)
        if not validator(generator):
            # See the validator implementations above for situations where
            # we will not attempt to generate SumTemplate objects
            return _TemplateIterManager.builtin_sum(generator, start=expr)
        return TemplateSumExpression((expr,), iters, _filter)


class _template_iter_manager:
    class _iter_wrapper:
        __slots__ = ('_class', '_iter', '_old_iter')

        def __init__(self, cls, context):
            def _iter_fcn(obj):
                return context.get_iter(obj)

            self._class = cls
            self._old_iter = cls.__iter__
            self._iter = _iter_fcn

        def acquire(self):
            self._class.__iter__ = self._iter

        def release(self):
            self._class.__iter__ = self._old_iter

    class _pause_template_iter_manager:
        __slots__ = ('iter_manager',)

        def __init__(self, iter_manager):
            self.iter_manager = iter_manager

        def __enter__(self):
            self.iter_manager.release()
            return self

        def __exit__(self, et, ev, tb):
            self.iter_manager.acquire()

    def __init__(self):
        self.paused = True
        self.context = None
        self.iters = None
        self.builtin_sum = builtins.sum

    def init(self, context, *iter_fcns):
        assert self.context is None
        self.context = context
        self.iters = [self._iter_wrapper(it, context) for it in iter_fcns]
        return self

    def acquire(self):
        assert self.paused
        self.paused = False
        builtins.sum = self.context.sum_template
        for it in self.iters:
            it.acquire()

    def release(self):
        assert not self.paused
        self.paused = True
        builtins.sum = self.builtin_sum
        for it in self.iters:
            it.release()

    def __enter__(self):
        assert self.context
        self.acquire()
        return self

    def __exit__(self, et, ev, tb):
        self.release()
        self.context = None
        self.iters = None

    def pause(self):
        if self.paused:
            return nullcontext()
        else:
            return self._pause_template_iter_manager(self)


# Global manager for coordinating overriding set iteration
_TemplateIterManager = _template_iter_manager()


def templatize_rule(block, rule, index_set):
    context = _template_iter_context()
    internal_error = None
    try:
        # Override Set iteration to return IndexTemplates
        with _TemplateIterManager.init(
            context,
            pyomo_core_base_set._FiniteSetMixin,
            GetItemExpression,
            GetAttrExpression,
        ):
            # Get the index templates needed for calling the rule
            if index_set is not None:
                # Note, do not rely on the __iter__ overload, as non-finite
                # Sets don't have an __iter__.
                indices = next(iter(context.get_iter(index_set)))
                try:
                    context.cache.pop()
                except IndexError:
                    assert indices is None
                    indices = ()
            else:
                indices = ()
            if type(indices) is not tuple:
                indices = (indices,)
            # Call the rule, returning the template expression and the
            # top-level IndexTemplate(s) generated when calling the rule.
            #
            # TBD: Should this just return a "FORALL()" expression node that
            # behaves similarly to the GetItemExpression node?
            return rule(block, indices), indices
    except:
        internal_error = sys.exc_info()
        raise
    finally:
        if len(context.cache):
            if internal_error is not None and not _SUPPRESS_TEMPLATIZATION_ERRORS:
                logger.error(
                    "The following exception was raised when "
                    "templatizing the rule '%s':\n\t%s"
                    % (
                        getattr(rule, '_fcn', rule.__class__).__name__,
                        internal_error[1],
                    )
                )
            raise TemplateExpressionError(
                None,
                "Explicit iteration (for loops) over Sets is not supported "
                "by template expressions.  Encountered loop over %s"
                % (context.cache[-1][0]._set,),
            )
    return None, indices


def _templatize_constraint_raw(con):
    """``templatize_rule`` for a constraint, with the tuple -> relational fix.

    Assumes any desired predicate-capture context is already active.
    """
    expr, indices = templatize_rule(con.parent_block(), con.rule, con.index_set())
    if expr.__class__ is tuple:
        expr = tuple_to_relational_expr(expr)
    return expr, indices


def templatize_constraint(con, capture_predicates=False):
    """Templatize a constraint family's rule.

    ``capture_predicates`` (Phase 3b, opt-in) enables index-predicate capture so
    a rule containing a filtered sum (``sum(... for j in J if j != n)``)
    templatizes into a filter-carrying ``TemplateSumExpression`` instead of
    failing.  Index conditionals still fall back (no ``conditional_policy``); use
    :func:`masked_templatize_constraint` for conditional cover.
    """
    if capture_predicates:
        with capture_index_predicates():
            return _templatize_constraint_raw(con)
    return _templatize_constraint_raw(con)


# Result-tag constants for masked_templatize_constraint.
MASKED_PLAIN = 'plain'
MASKED_CONDITIONAL = 'masked'
_MAX_CONDITIONAL_PREDICATES = 6


def masked_templatize_constraint(con):
    """Templatize a rule that contains index conditionals, via polarity replay.

    An index conditional -- ``(m.s[n, t - 1] if t > 0 else 0)`` or a guarded
    ``if t == 0: return Constraint.Skip`` -- selects a *branch* by a predicate on
    the row index.  Python evaluates only the taken branch, so a single
    templatization cannot see both.  We therefore *replay* the rule once per
    truth-assignment (polarity combination) of the conditional predicates it
    contains, each replay forcing those predicates to fixed booleans and yielding
    an ordinary (branch-free) template; at extraction each row uses the template
    of the combination its index actually satisfies (and is dropped if that
    combination returns ``Constraint.Skip``).  Filtered sums inside a branch are
    captured as usual, so conditionals and filters compose.

    Returns one of:

    * ``(MASKED_PLAIN, (expr, indices))`` -- no index conditionals were found
      (the rule templatized with filters only); the caller stores it as a plain
      template.
    * ``(MASKED_CONDITIONAL, (row_templates, predicates, combos))`` -- where
      ``predicates`` is the ordered list of conditional predicates (over
      ``row_templates``) and ``combos`` maps each truth-assignment tuple to a
      plain ``(expr, indices)`` template or to ``Constraint.Skip``.

    Raises (caller falls back to classic) if the conditional structure is not a
    flat, index-only set of predicates -- e.g. a predicate that is not a pure
    index comparison, more than :data:`_MAX_CONDITIONAL_PREDICATES` predicates,
    or a structure whose predicate set changes with the branch taken (nested /
    data-dependent conditionals).
    """
    from pyomo.core.base.indexed_component import IndexedComponent

    Skip = IndexedComponent.Skip

    # --- Pass 1: discovery (force every conditional False) ---------------- #
    discovered = []

    def discover(expr, cap):
        discovered.append(expr)
        return False

    with capture_index_predicates(conditional_policy=discover) as cap:
        info0 = _templatize_constraint_raw(con)
        if cap.unsupported:
            raise TemplateExpressionError(
                None, "index conditional predicate is not vectorizable"
            )
    k = len(discovered)
    if k == 0:
        return MASKED_PLAIN, info0
    if k > _MAX_CONDITIONAL_PREDICATES:
        raise TemplateExpressionError(
            None, "too many index conditionals to vectorize (%d)" % (k,)
        )
    pred_signature = [p.to_string() for p in discovered]
    row_templates = info0[1]

    # --- Replay each polarity combination -------------------------------- #
    combos = {(False,) * k: info0}
    for bits in itertools.product((False, True), repeat=k):
        if bits in combos:
            continue
        replay = {'preds': [], 'i': 0}

        def force(expr, cap, _bits=bits, _r=replay):
            _r['preds'].append(expr)
            j = _r['i']
            _r['i'] += 1
            return _bits[j] if j < len(_bits) else False

        with capture_index_predicates(conditional_policy=force) as cap:
            infob = _templatize_constraint_raw(con)
            if (
                cap.unsupported
                or replay['i'] != k
                or [p.to_string() for p in replay['preds']] != pred_signature
            ):
                raise TemplateExpressionError(
                    None, "inconsistent index-conditional structure"
                )
        combos[bits] = infob
    return MASKED_CONDITIONAL, (row_templates, discovered, combos)


def _scalar_predicate_value(pred):
    """Evaluate one index predicate (templates already set) to a Python bool.

    Note we evaluate *by value* rather than ``bool(pred)``: a NotEqualExpression's
    ``__bool__`` short-circuits on object identity, which is not the index-value
    comparison we need here.
    """
    if pred.__class__ is EqualityExpression:
        return value(pred.args[0]) == value(pred.args[1])
    if pred.__class__ is NotEqualExpression:
        return value(pred.args[0]) != value(pred.args[1])
    if pred.__class__ is InequalityExpression:
        a, b = value(pred.args[0]), value(pred.args[1])
        return a < b if pred.strict else a <= b
    if pred.__class__ is RangedExpression:
        lb, x, ub = (value(a) for a in pred.args)
        s = pred.strict
        return (lb < x if s[0] else lb <= x) and (x < ub if s[1] else x <= ub)
    raise TemplateExpressionError(None, "unsupported index predicate")


def evaluate_index_predicates(predicates, row_templates, index):
    """Return the truth-assignment tuple of ``predicates`` at one row ``index``.

    Sets the row IndexTemplates to the concrete index values and evaluates each
    predicate.  This is a cheap O(1)-per-predicate scalar evaluation (no
    expression-tree build, no template resolution of the body) used at construct
    time to route each row to its polarity combination.
    """
    if index.__class__ is not tuple:
        index = (index,)
    for t, v in zip(row_templates, index):
        t.set_value(v)
    return tuple(_scalar_predicate_value(p) for p in predicates)
