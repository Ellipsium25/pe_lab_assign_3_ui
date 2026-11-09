from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class StreamDef:
    id: str
    desc: str = ""


@dataclass
class StreamInput:
    total_flow: Optional[float] = None
    mode: str = "mole_fraction"  # "mole_fraction" | "component_flow"
    values: List[Optional[float]] = field(default_factory=list)


@dataclass
class MaterialBalanceProblem:
    species: List[str]
    inlet_streams: List[StreamDef]
    outlet_streams: List[StreamDef]
    reactions: List[List[float]]  # shape: (N_r, N_c), negative=reactant, positive=product
    stream_inputs: Dict[str, StreamInput] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    flow_unit: str = "kmol/h"

    @property
    def n_c(self) -> int:
        return len(self.species)

    @property
    def n_in(self) -> int:
        return len(self.inlet_streams)

    @property
    def n_out(self) -> int:
        return len(self.outlet_streams)

    @property
    def n_r(self) -> int:
        return len(self.reactions)

    @property
    def all_streams(self) -> List[StreamDef]:
        return self.inlet_streams + self.outlet_streams

    @property
    def n_streams(self) -> int:
        return len(self.all_streams)

    @property
    def n_vars(self) -> int:
        return self.n_streams * self.n_c + self.n_r


@dataclass
class DOFReport:
    n_vars: int
    n_material_balances: int
    n_user_specs: int
    n_total_equations: int
    jacobian_rank: int
    effective_dof: int
    redundant_count: int
    is_linear: bool
    status: str  # "WELL_POSED" | "UNDER_SPECIFIED" | "OVER_SPECIFIED" | "INVALID"
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    equation_labels: List[str] = field(default_factory=list)


@dataclass
class SolutionResult:
    success: bool
    status: str  # "SOLVED" | "SOLVED_WITH_WARNINGS" | "UNDER_SPECIFIED" | "CONTRADICTORY" | "FAILED"
    message: str
    dof: DOFReport
    component_flows: Optional[np.ndarray] = None  # shape (N_streams, N_c)
    total_flows: Optional[np.ndarray] = None      # shape (N_streams,)
    mole_fractions: Optional[np.ndarray] = None   # shape (N_streams, N_c)
    extents: Optional[np.ndarray] = None          # shape (N_r,)
    max_residual: float = 0.0
    stream_table_df: Optional[pd.DataFrame] = None
    component_flow_df: Optional[pd.DataFrame] = None
    balance_check_df: Optional[pd.DataFrame] = None
    reaction_summary_df: Optional[pd.DataFrame] = None
    species_conversion_df: Optional[pd.DataFrame] = None
    equation_residuals: List[Tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_ident(name: str) -> str:
    """Convert species/stream names into safe alphanumeric+underscore tokens for custom equations."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")
    return cleaned or "item"


def format_reaction_str(species: List[str], coeffs: List[float]) -> str:
    reactants = []
    products = []
    for sp_name, nu in zip(species, coeffs):
        if abs(nu) < 1e-12:
            continue
        mag = abs(nu)
        mag_str = "" if abs(mag - 1.0) < 1e-9 else f"{mag:g} "
        if nu < 0:
            reactants.append(f"{mag_str}{sp_name}")
        else:
            products.append(f"{mag_str}{sp_name}")
    lhs = " + ".join(reactants) if reactants else "0"
    rhs = " + ".join(products) if products else "0"
    return f"{lhs} → {rhs}"


def get_custom_symbol_guide(problem: MaterialBalanceProblem) -> Dict[str, List[str]]:
    stream_tokens = [sanitize_ident(s.id) for s in problem.all_streams]
    species_tokens = [sanitize_ident(sp_name) for sp_name in problem.species]
    return {
        "total_flows": [f"F_{st_tok}" for st_tok in stream_tokens],
        "component_flows": [
            f"n_{st_tok}_{sp_tok}" for st_tok in stream_tokens for sp_tok in species_tokens
        ],
        "mole_fractions": [
            f"x_{st_tok}_{sp_tok}" for st_tok in stream_tokens for sp_tok in species_tokens
        ],
        "extents": [f"xi_{m + 1}" for m in range(problem.n_r)],
    }


# ---------------------------------------------------------------------------
# Equation Builder
# ---------------------------------------------------------------------------

class EquationSystem:
    def __init__(self, problem: MaterialBalanceProblem):
        self.problem = problem
        self.n_c = problem.n_c
        self.n_s = problem.n_streams
        self.n_r = problem.n_r
        self.n_vars = problem.n_vars

        self.stream_index = {s.id: i for i, s in enumerate(problem.all_streams)}
        self.species_index = {sp_name: i for i, sp_name in enumerate(problem.species)}

        self.linear_rows: List[Tuple[np.ndarray, float, str]] = []
        self.nonlinear_eqs: List[Tuple[Callable[[np.ndarray], float], str]] = []
        self.validation_errors: List[str] = []
        self.warnings: List[str] = []
        self.typical_scale: float = 100.0

        self._validate_structure()
        if not self.validation_errors:
            self._build_material_balances()
            self._build_stream_specs()
            self._build_constraints()

    def _idx_n(self, s_idx: int, k_idx: int) -> int:
        return s_idx * self.n_c + k_idx

    def _idx_xi(self, m_idx: int) -> int:
        return self.n_s * self.n_c + m_idx

    def _validate_structure(self) -> None:
        if self.n_c < 1:
            self.validation_errors.append("Number of chemical species (N_c) must be at least 1.")
        if len(self.problem.inlet_streams) < 1:
            self.validation_errors.append("At least 1 inlet stream (N_in) is required.")
        if len(self.problem.outlet_streams) < 1:
            self.validation_errors.append("At least 1 outlet stream (N_out) is required.")

        clean_species = [s.strip() for s in self.problem.species]
        if any(not s for s in clean_species):
            self.validation_errors.append("All species must have non-empty names.")
        if len(set(clean_species)) != len(clean_species):
            self.validation_errors.append("Species names must be unique.")

        stream_ids = [s.id.strip() for s in self.problem.all_streams]
        if any(not sid for sid in stream_ids):
            self.validation_errors.append("All streams must have non-empty IDs.")
        if len(set(stream_ids)) != len(stream_ids):
            self.validation_errors.append("All stream IDs (inlet + outlet) must be unique.")

        for m, coeffs in enumerate(self.problem.reactions):
            if len(coeffs) != self.n_c:
                self.validation_errors.append(
                    f"Reaction {m + 1} has {len(coeffs)} coefficients, expected {self.n_c}."
                )
            elif all(abs(c) < 1e-12 for c in coeffs):
                self.validation_errors.append(
                    f"Reaction {m + 1} has all-zero stoichiometric coefficients."
                )

    def _build_material_balances(self) -> None:
        n_in = self.problem.n_in
        for k, sp_name in enumerate(self.problem.species):
            row = np.zeros(self.n_vars, dtype=float)
            for s_idx in range(n_in):
                row[self._idx_n(s_idx, k)] += 1.0
            for s_idx in range(n_in, self.n_s):
                row[self._idx_n(s_idx, k)] -= 1.0
            for m in range(self.n_r):
                row[self._idx_xi(m)] += float(self.problem.reactions[m][k])
            self.linear_rows.append((row, 0.0, f"Material Balance ({sp_name})"))

    def _build_stream_specs(self) -> None:
        observed_flows: List[float] = []

        for s_idx, stream in enumerate(self.problem.all_streams):
            spec = self.problem.stream_inputs.get(stream.id)
            if spec is None:
                continue

            vals = spec.values if spec.values else [None] * self.n_c
            if len(vals) != self.n_c:
                self.validation_errors.append(
                    f"Stream '{stream.id}' input count ({len(vals)}) does not match N_c ({self.n_c})."
                )
                continue

            F_spec = spec.total_flow
            if F_spec is not None:
                if F_spec < 0:
                    self.validation_errors.append(
                        f"Stream '{stream.id}' total flow cannot be negative ({F_spec})."
                    )
                    continue
                if F_spec > 0:
                    observed_flows.append(F_spec)

            if spec.mode == "component_flow":
                for k, v in enumerate(vals):
                    if v is not None and v < 0:
                        self.validation_errors.append(
                            f"Component flow of {self.problem.species[k]} in '{stream.id}' cannot be negative."
                        )
                if F_spec is not None:
                    row = np.zeros(self.n_vars, dtype=float)
                    for k in range(self.n_c):
                        row[self._idx_n(s_idx, k)] = 1.0
                    self.linear_rows.append((row, float(F_spec), f"Total Flow F_{stream.id} = {F_spec:g}"))

                for k, val in enumerate(vals):
                    if val is not None:
                        if val > 0:
                            observed_flows.append(val)
                        row = np.zeros(self.n_vars, dtype=float)
                        row[self._idx_n(s_idx, k)] = 1.0
                        self.linear_rows.append(
                            (row, float(val), f"Component Flow n_{stream.id}_{self.problem.species[k]} = {val:g}")
                        )

            elif spec.mode == "mole_fraction":
                given_indices = [k for k, v in enumerate(vals) if v is not None]
                for k in given_indices:
                    x_val = float(vals[k])  # type: ignore[arg-type]
                    if x_val < -1e-9 or x_val > 1.0 + 1e-9:
                        self.validation_errors.append(
                            f"Mole fraction of {self.problem.species[k]} in '{stream.id}' must be between 0 and 1 (got {x_val})."
                        )

                x_sum = sum(float(vals[k]) for k in given_indices)  # type: ignore[arg-type]
                if len(given_indices) == self.n_c:
                    if abs(x_sum - 1.0) > 1e-3:
                        self.validation_errors.append(
                            f"Mole fractions in stream '{stream.id}' sum to {x_sum:.4f}, but must sum to 1.0."
                        )
                        continue
                elif x_sum > 1.0 + 1e-3:
                    self.validation_errors.append(
                        f"Entered mole fractions in stream '{stream.id}' already exceed 1.0 (sum = {x_sum:.4f})."
                    )
                    continue

                if F_spec is not None:
                    row = np.zeros(self.n_vars, dtype=float)
                    for k in range(self.n_c):
                        row[self._idx_n(s_idx, k)] = 1.0
                    self.linear_rows.append((row, float(F_spec), f"Total Flow F_{stream.id} = {F_spec:g}"))

                    if len(given_indices) == self.n_c and self.n_c > 0:
                        skip_k = max(given_indices, key=lambda idx: float(vals[idx]))  # type: ignore[arg-type]
                        active_indices = [k for k in given_indices if k != skip_k]
                    else:
                        active_indices = given_indices

                    for k in active_indices:
                        x_val = float(vals[k])  # type: ignore[arg-type]
                        row = np.zeros(self.n_vars, dtype=float)
                        row[self._idx_n(s_idx, k)] = 1.0
                        self.linear_rows.append(
                            (
                                row,
                                float(F_spec) * x_val,
                                f"Composition n_{stream.id}_{self.problem.species[k]} = {F_spec:g} * {x_val:g}",
                            )
                        )
                else:
                    if len(given_indices) == self.n_c and self.n_c > 0:
                        skip_k = max(given_indices, key=lambda idx: float(vals[idx]))  # type: ignore[arg-type]
                        active_indices = [k for k in given_indices if k != skip_k]
                    else:
                        active_indices = given_indices

                    for k in active_indices:
                        x_val = float(vals[k])  # type: ignore[arg-type]
                        row = np.zeros(self.n_vars, dtype=float)
                        for c in range(self.n_c):
                            row[self._idx_n(s_idx, c)] -= x_val
                        row[self._idx_n(s_idx, k)] += 1.0
                        self.linear_rows.append(
                            (
                                row,
                                0.0,
                                f"Mole Fraction x_{stream.id}_{self.problem.species[k]} = {x_val:g}",
                            )
                        )
            else:
                self.validation_errors.append(f"Unknown input mode '{spec.mode}' for stream '{stream.id}'.")

        if observed_flows:
            self.typical_scale = max(observed_flows)

    def _build_constraints(self) -> None:
        n_in = self.problem.n_in
        for idx, c in enumerate(self.problem.constraints):
            ctype = c.get("type")

            if ctype == "conversion":
                sp_name = c.get("species", "")
                if sp_name not in self.species_index:
                    self.validation_errors.append(f"Constraint {idx + 1}: unknown species '{sp_name}'.")
                    continue
                X_val = float(c.get("value", 0.0))
                if X_val < 0.0 or X_val > 1.0:
                    self.validation_errors.append(
                        f"Conversion of {sp_name} must be fractional between 0 and 1 (got {X_val})."
                    )
                    continue
                k = self.species_index[sp_name]
                row = np.zeros(self.n_vars, dtype=float)
                for s_idx in range(n_in):
                    row[self._idx_n(s_idx, k)] = -(1.0 - X_val)
                for s_idx in range(n_in, self.n_s):
                    row[self._idx_n(s_idx, k)] = 1.0
                self.linear_rows.append((row, 0.0, f"Conversion X_{sp_name} = {X_val:.4g}"))

            elif ctype == "extent":
                rxn_num = int(c.get("reaction_idx", 1))
                if rxn_num < 1 or rxn_num > self.n_r:
                    self.validation_errors.append(
                        f"Constraint {idx + 1}: invalid reaction index {rxn_num}."
                    )
                    continue
                val = float(c.get("value", 0.0))
                row = np.zeros(self.n_vars, dtype=float)
                row[self._idx_xi(rxn_num - 1)] = 1.0
                self.linear_rows.append((row, val, f"Reaction {rxn_num} Extent xi_{rxn_num} = {val:g}"))

            elif ctype == "split":
                sp_name = c.get("species", "")
                sid = c.get("stream_id", "")
                basis = c.get("basis", "outlet")
                frac = float(c.get("fraction", 0.0))
                if sp_name not in self.species_index or sid not in self.stream_index:
                    self.validation_errors.append(f"Constraint {idx + 1}: invalid species or stream in split.")
                    continue
                if frac < 0.0 or frac > 1.0:
                    self.validation_errors.append(
                        f"Split fraction for {sp_name} in {sid} must be between 0 and 1 (got {frac})."
                    )
                    continue
                k = self.species_index[sp_name]
                target_s = self.stream_index[sid]
                if target_s < n_in:
                    self.validation_errors.append(
                        f"Split target stream '{sid}' must be an outlet stream."
                    )
                    continue
                row = np.zeros(self.n_vars, dtype=float)
                if basis == "feed":
                    for s_idx in range(n_in):
                        row[self._idx_n(s_idx, k)] -= frac
                    row[self._idx_n(target_s, k)] += 1.0
                    label = f"Recovery n_{sid}_{sp_name} = {frac:g} * Feed({sp_name})"
                else:
                    for s_idx in range(n_in, self.n_s):
                        row[self._idx_n(s_idx, k)] -= frac
                    row[self._idx_n(target_s, k)] += 1.0
                    label = f"Outlet Split n_{sid}_{sp_name} = {frac:g} * Out({sp_name})"
                self.linear_rows.append((row, 0.0, label))

            elif ctype == "flow_ratio":
                s1 = c.get("num_stream", "")
                s2 = c.get("den_stream", "")
                ratio = float(c.get("ratio", 1.0))
                if s1 not in self.stream_index or s2 not in self.stream_index:
                    self.validation_errors.append(f"Constraint {idx + 1}: unknown stream in flow ratio.")
                    continue
                if s1 == s2:
                    self.validation_errors.append("Flow ratio requires two different streams.")
                    continue
                i1, i2 = self.stream_index[s1], self.stream_index[s2]
                row = np.zeros(self.n_vars, dtype=float)
                for k in range(self.n_c):
                    row[self._idx_n(i1, k)] += 1.0
                    row[self._idx_n(i2, k)] -= ratio
                self.linear_rows.append((row, 0.0, f"Flow Ratio F_{s1} = {ratio:g} * F_{s2}"))

            elif ctype == "molar_ratio":
                sid = c.get("stream_id", "")
                sp1 = c.get("num_species", "")
                sp2 = c.get("den_species", "")
                ratio = float(c.get("ratio", 1.0))
                if sid not in self.stream_index or sp1 not in self.species_index or sp2 not in self.species_index:
                    self.validation_errors.append(f"Constraint {idx + 1}: unknown stream/species in molar ratio.")
                    continue
                if sp1 == sp2:
                    self.validation_errors.append("Molar ratio requires two different species.")
                    continue
                s_idx = self.stream_index[sid]
                k1, k2 = self.species_index[sp1], self.species_index[sp2]
                row = np.zeros(self.n_vars, dtype=float)
                row[self._idx_n(s_idx, k1)] += 1.0
                row[self._idx_n(s_idx, k2)] -= ratio
                self.linear_rows.append((row, 0.0, f"Molar Ratio ({sid}): n_{sp1} = {ratio:g} * n_{sp2}"))

            elif ctype == "equilibrium":
                rxn_num = int(c.get("reaction_idx", 1))
                sid = c.get("stream_id", "")
                keq = float(c.get("keq", 1.0))
                pressure = float(c.get("pressure", 1.0))
                if rxn_num < 1 or rxn_num > self.n_r:
                    self.validation_errors.append(f"Equilibrium constraint: invalid reaction {rxn_num}.")
                    continue
                if sid not in self.stream_index or self.stream_index[sid] < n_in:
                    self.validation_errors.append(
                        f"Equilibrium stream '{sid}' must be an outlet stream."
                    )
                    continue
                if keq <= 0 or pressure <= 0:
                    self.validation_errors.append("Equilibrium constant K_eq and pressure P must be positive.")
                    continue

                s_idx = self.stream_index[sid]
                coeffs = self.problem.reactions[rxn_num - 1]
                s_plus = sum(nu for nu in coeffs if nu > 0)
                s_minus = sum(-nu for nu in coeffs if nu < 0)
                delta_nu = s_plus - s_minus
                kx = keq * (pressure ** (-delta_nu))

                p_plus = max(0.0, -delta_nu)
                p_minus = max(0.0, delta_nu)
                scale_norm = max(1.0, self.typical_scale) ** max(s_plus + p_plus, s_minus + p_minus, 1.0)
                offset = s_idx * self.n_c

                def make_keq_func(
                    off: int,
                    nc: int,
                    nu_list: List[float],
                    pp: float,
                    pm: float,
                    k_x: float,
                    norm: float,
                ) -> Callable[[np.ndarray], float]:
                    def eq_fn(z: np.ndarray) -> float:
                        n_s = np.maximum(z[off : off + nc], 0.0)
                        F_s = float(np.sum(n_s))
                        prod_term = 1.0
                        react_term = 1.0
                        for k_i, nu in enumerate(nu_list):
                            if nu > 0:
                                prod_term *= float(n_s[k_i]) ** nu
                            elif nu < 0:
                                react_term *= float(n_s[k_i]) ** (-nu)
                        lhs = prod_term * (F_s ** pp if pp > 0 else 1.0)
                        rhs = k_x * react_term * (F_s ** pm if pm > 0 else 1.0)
                        return (lhs - rhs) / norm
                    return eq_fn

                self.nonlinear_eqs.append(
                    (
                        make_keq_func(offset, self.n_c, list(coeffs), p_plus, p_minus, kx, scale_norm),
                        f"Equilibrium R{rxn_num} in {sid} (K_eq={keq:g}, P={pressure:g})",
                    )
                )

            elif ctype == "custom":
                eq_str = str(c.get("equation", "")).strip()
                if not eq_str:
                    continue
                self._parse_custom_equation(eq_str)

            else:
                self.validation_errors.append(f"Unknown constraint type '{ctype}'.")

    def _parse_custom_equation(self, eq_str: str) -> None:
        if "__" in eq_str:
            self.validation_errors.append(f"Invalid custom equation '{eq_str}': '__' is not allowed.")
            return

        z_syms = [sp.Symbol(f"__z_{i}") for i in range(self.n_vars)]
        local_map: Dict[str, Any] = {
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "Abs": sp.Abs,
        }

        stream_tok_map = {sanitize_ident(s.id): idx for idx, s in enumerate(self.problem.all_streams)}
        species_tok_map = {sanitize_ident(name): idx for idx, name in enumerate(self.problem.species)}

        for st_tok, s_idx in stream_tok_map.items():
            F_expr = sum(z_syms[self._idx_n(s_idx, k)] for k in range(self.n_c))
            local_map[f"F_{st_tok}"] = F_expr
            for sp_tok, k_idx in species_tok_map.items():
                n_sym = z_syms[self._idx_n(s_idx, k_idx)]
                local_map[f"n_{st_tok}_{sp_tok}"] = n_sym
                local_map[f"x_{st_tok}_{sp_tok}"] = n_sym / (F_expr + 1e-12)

        for m in range(self.n_r):
            local_map[f"xi_{m + 1}"] = z_syms[self._idx_xi(m)]

        safe_globals: Dict[str, Any] = {
            "__builtins__": {},
            "Symbol": sp.Symbol,
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Rational": sp.Rational,
            "factorial": sp.factorial,
            "Lambda": sp.Lambda,
        }
        transformations = standard_transformations + (convert_xor,)
        try:
            if "=" in eq_str:
                parts = eq_str.split("=")
                if len(parts) != 2:
                    raise ValueError("Use exactly one '=' sign in an equation.")
                lhs = parse_expr(
                    parts[0],
                    local_dict=local_map,
                    global_dict=safe_globals,
                    transformations=transformations,
                )
                rhs = parse_expr(
                    parts[1],
                    local_dict=local_map,
                    global_dict=safe_globals,
                    transformations=transformations,
                )
                expr = sp.simplify(lhs - rhs)
            else:
                expr = parse_expr(
                    eq_str,
                    local_dict=local_map,
                    global_dict=safe_globals,
                    transformations=transformations,
                )

            unknown = expr.free_symbols - set(z_syms)
            if unknown:
                names = ", ".join(sorted(str(s) for s in unknown))
                raise ValueError(f"Unrecognized symbol(s): {names}")

            # Try converting to a linear matrix row if strictly linear in z_syms
            try:
                A_mat, b_vec = sp.linear_eq_to_matrix([expr], z_syms)
                row = np.array([float(A_mat[0, j]) for j in range(self.n_vars)], dtype=float)
                rhs_val = float(b_vec[0, 0])
                self.linear_rows.append((row, rhs_val, f"Custom: {eq_str}"))
                return
            except Exception:
                pass

            fn_raw = sp.lambdify(z_syms, expr, modules="numpy")

            def make_custom_fn(fn: Callable[..., Any]) -> Callable[[np.ndarray], float]:
                def wrapped(z: np.ndarray) -> float:
                    return float(fn(*z))
                return wrapped

            self.nonlinear_eqs.append((make_custom_fn(fn_raw), f"Custom: {eq_str}"))
        except Exception as exc:
            self.validation_errors.append(f"Invalid custom equation '{eq_str}': {exc}")

    # -----------------------------------------------------------------------
    # Evaluation & Rank Analysis
    # -----------------------------------------------------------------------

    def residual_vector(self, z: np.ndarray) -> np.ndarray:
        res: List[float] = []
        for row, rhs, _ in self.linear_rows:
            res.append(float(np.dot(row, z) - rhs))
        for fn, _ in self.nonlinear_eqs:
            res.append(float(fn(z)))
        return np.array(res, dtype=float)

    def equation_labels(self) -> List[str]:
        return [lbl for _, _, lbl in self.linear_rows] + [lbl for _, lbl in self.nonlinear_eqs]

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        m_lin = len(self.linear_rows)
        m_nl = len(self.nonlinear_eqs)
        m_tot = m_lin + m_nl
        J = np.zeros((m_tot, self.n_vars), dtype=float)
        if m_lin > 0:
            J[:m_lin, :] = np.vstack([row for row, _, _ in self.linear_rows])
        for i, (fn, _) in enumerate(self.nonlinear_eqs):
            row_idx = m_lin + i
            for j in range(self.n_vars):
                eps = 1e-6 * max(1.0, abs(z[j]))
                zp = z.copy()
                zm = z.copy()
                zp[j] += eps
                zm[j] -= eps
                J[row_idx, j] = (fn(zp) - fn(zm)) / (2.0 * eps)
        return J

    def analyze_dof(self) -> DOFReport:
        labels = self.equation_labels()
        n_mb = self.n_c
        n_tot_eq = len(labels)
        n_user = max(0, n_tot_eq - n_mb)

        if self.validation_errors:
            return DOFReport(
                n_vars=self.n_vars,
                n_material_balances=n_mb,
                n_user_specs=n_user,
                n_total_equations=n_tot_eq,
                jacobian_rank=0,
                effective_dof=self.n_vars,
                redundant_count=0,
                is_linear=(len(self.nonlinear_eqs) == 0),
                status="INVALID",
                validation_errors=list(self.validation_errors),
                warnings=list(self.warnings),
                equation_labels=labels,
            )

        rng = np.random.default_rng(42)
        z_probe = rng.uniform(0.7, 2.3, size=self.n_vars) * max(1.0, self.typical_scale * 0.1)
        J = self.jacobian(z_probe)
        rank = int(np.linalg.matrix_rank(J, tol=1e-8)) if J.size > 0 else 0

        eff_dof = self.n_vars - rank
        redundant = n_tot_eq - rank
        warnings = list(self.warnings)

        if redundant > 0:
            warnings.append(
                f"{redundant} equation(s) appear linearly dependent or redundant with other specifications."
            )

        if eff_dof > 0:
            status = "UNDER_SPECIFIED"
        elif n_tot_eq > rank:
            status = "OVER_SPECIFIED"
        else:
            status = "WELL_POSED"

        return DOFReport(
            n_vars=self.n_vars,
            n_material_balances=n_mb,
            n_user_specs=n_user,
            n_total_equations=n_tot_eq,
            jacobian_rank=rank,
            effective_dof=eff_dof,
            redundant_count=redundant,
            is_linear=(len(self.nonlinear_eqs) == 0),
            status=status,
            validation_errors=[],
            warnings=warnings,
            equation_labels=labels,
        )


# ---------------------------------------------------------------------------
# Main Solve Function
# ---------------------------------------------------------------------------

def solve_material_balance(problem: MaterialBalanceProblem) -> SolutionResult:
    sys = EquationSystem(problem)
    dof = sys.analyze_dof()

    if dof.status == "INVALID":
        return SolutionResult(
            success=False,
            status="FAILED",
            message="Validation failed: " + " | ".join(dof.validation_errors),
            dof=dof,
        )

    if dof.status == "UNDER_SPECIFIED":
        return SolutionResult(
            success=False,
            status="UNDER_SPECIFIED",
            message=(
                f"System is under-specified (Effective DOF = {dof.effective_dof}). "
                f"Provide {dof.effective_dof} more independent specification(s)."
            ),
            dof=dof,
        )

    z_sol: Optional[np.ndarray] = None
    if dof.is_linear:
        A = np.vstack([row for row, _, _ in sys.linear_rows])
        b = np.array([rhs for _, rhs, _ in sys.linear_rows], dtype=float)
        if dof.status == "WELL_POSED" and A.shape[0] == A.shape[1]:
            z_sol = np.linalg.solve(A, b)
        else:
            z_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    else:
        n_comp_vars = problem.n_streams * problem.n_c
        lb = np.concatenate([np.zeros(n_comp_vars), np.full(problem.n_r, -np.inf)])
        ub = np.full(problem.n_vars, np.inf)

        scale = max(1.0, sys.typical_scale)
        starts: List[np.ndarray] = []

        if sys.linear_rows:
            A_lin = np.vstack([row for row, _, _ in sys.linear_rows])
            b_lin = np.array([rhs for _, rhs, _ in sys.linear_rows], dtype=float)
            z_lin, _, _, _ = np.linalg.lstsq(A_lin, b_lin, rcond=None)
            z_lin_clipped = z_lin.copy()
            z_lin_clipped[:n_comp_vars] = np.maximum(z_lin_clipped[:n_comp_vars], 0.05 * scale)
            starts.append(z_lin_clipped)

        starts.append(np.concatenate([np.full(n_comp_vars, 0.5 * scale), np.zeros(problem.n_r)]))
        starts.append(np.concatenate([np.full(n_comp_vars, 0.25 * scale), np.full(problem.n_r, 0.1 * scale)]))

        best_res_norm = np.inf
        for z0 in starts:
            res = least_squares(
                sys.residual_vector,
                z0,
                bounds=(lb, ub),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=5000,
            )
            norm = float(np.max(np.abs(res.fun))) if res.fun.size else 0.0
            if norm < best_res_norm:
                best_res_norm = norm
                z_sol = res.x
            if norm < 1e-9:
                break

    assert z_sol is not None
    residuals = sys.residual_vector(z_sol)
    labels = sys.equation_labels()
    eq_residuals = list(zip(labels, [float(r) for r in residuals]))
    max_res = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    tol = max(1e-5, 1e-6 * sys.typical_scale)

    if max_res > tol:
        worst_lbl, worst_val = max(eq_residuals, key=lambda x: abs(x[1]))
        return SolutionResult(
            success=False,
            status="CONTRADICTORY",
            message=(
                f"Equations could not be simultaneously satisfied (max residual = {max_res:.3e} "
                f"at '{worst_lbl}': {worst_val:.3e}). Check for contradictory specifications."
            ),
            dof=dof,
            max_residual=max_res,
            equation_residuals=eq_residuals,
        )

    n_comp = problem.n_streams * problem.n_c
    comp_flows = z_sol[:n_comp].reshape((problem.n_streams, problem.n_c))
    comp_flows = np.where(np.abs(comp_flows) < 1e-10, 0.0, comp_flows)
    extents = z_sol[n_comp:] if problem.n_r > 0 else np.zeros(0)

    neg_warnings: List[str] = []
    for s_idx, st_def in enumerate(problem.all_streams):
        for k, sp_name in enumerate(problem.species):
            if comp_flows[s_idx, k] < -1e-6:
                neg_warnings.append(
                    f"Negative molar flow calculated for {sp_name} in '{st_def.id}' ({comp_flows[s_idx, k]:.4g})."
                )

    total_flows = np.sum(comp_flows, axis=1)
    mole_fractions = np.zeros_like(comp_flows)
    for s_idx in range(problem.n_streams):
        if abs(total_flows[s_idx]) > 1e-12:
            mole_fractions[s_idx, :] = comp_flows[s_idx, :] / total_flows[s_idx]

    stream_rows = []
    comp_rows = []
    for s_idx, st_def in enumerate(problem.all_streams):
        stype = "Inlet" if s_idx < problem.n_in else "Outlet"
        row_mf: Dict[str, Any] = {
            "Stream": st_def.id,
            "Type": stype,
            "Description": st_def.desc,
            f"Total Flow ({problem.flow_unit})": round(float(total_flows[s_idx]), 4),
        }
        row_cf: Dict[str, Any] = {
            "Stream": st_def.id,
            "Type": stype,
            "Description": st_def.desc,
            f"Total Flow ({problem.flow_unit})": round(float(total_flows[s_idx]), 4),
        }
        for k, sp_name in enumerate(problem.species):
            row_mf[f"x_{sp_name}"] = round(float(mole_fractions[s_idx, k]), 4)
            row_cf[sp_name] = round(float(comp_flows[s_idx, k]), 4)
        stream_rows.append(row_mf)
        comp_rows.append(row_cf)

    stream_table_df = pd.DataFrame(stream_rows)
    component_flow_df = pd.DataFrame(comp_rows)

    n_in = problem.n_in
    bal_rows = []
    for k, sp_name in enumerate(problem.species):
        f_in = float(np.sum(comp_flows[:n_in, k]))
        f_out = float(np.sum(comp_flows[n_in:, k]))
        gen = sum(float(problem.reactions[m][k]) * float(extents[m]) for m in range(problem.n_r))
        residual = f_in - f_out + gen
        bal_rows.append(
            {
                "Species": sp_name,
                f"Total In ({problem.flow_unit})": round(f_in, 4),
                f"Net Generation ({problem.flow_unit})": round(gen, 4),
                f"Total Out ({problem.flow_unit})": round(f_out, 4),
                "Balance Residual": float(f"{residual:.2e}"),
            }
        )
    balance_check_df = pd.DataFrame(bal_rows)

    rxn_rows = []
    for m in range(problem.n_r):
        rxn_rows.append(
            {
                "Reaction": f"R{m + 1}: {format_reaction_str(problem.species, problem.reactions[m])}",
                f"Extent ξ ({problem.flow_unit})": round(float(extents[m]), 4),
            }
        )
    reaction_summary_df = pd.DataFrame(rxn_rows)

    conv_rows = []
    for k, sp_name in enumerate(problem.species):
        f_in = float(np.sum(comp_flows[:n_in, k]))
        f_out = float(np.sum(comp_flows[n_in:, k]))
        if f_in > 1e-12 and f_out <= f_in + 1e-9:
            conv = (f_in - f_out) / f_in * 100.0
            conv_str = f"{conv:.2f}%"
        elif f_in <= 1e-12:
            conv_str = "— (not fed)"
        else:
            conv_str = "Net Produced"
        conv_rows.append(
            {
                "Species": sp_name,
                f"Fed ({problem.flow_unit})": round(f_in, 4),
                f"Exit ({problem.flow_unit})": round(f_out, 4),
                "Overall Conversion": conv_str,
            }
        )
    species_conversion_df = pd.DataFrame(conv_rows)

    all_warnings = dof.warnings + neg_warnings
    if neg_warnings:
        status = "SOLVED_WITH_WARNINGS"
        msg = "Solved mathematically, but physically impossible negative flows occurred. Check your specs."
    elif all_warnings:
        status = "SOLVED_WITH_WARNINGS"
        msg = "All material balance equations are satisfied (with redundant consistent specs)."
    else:
        status = "SOLVED"
        msg = "Solution completed. All material balance equations are satisfied."

    return SolutionResult(
        success=(len(neg_warnings) == 0),
        status=status,
        message=msg,
        dof=dof,
        component_flows=comp_flows,
        total_flows=total_flows,
        mole_fractions=mole_fractions,
        extents=extents,
        max_residual=max_res,
        stream_table_df=stream_table_df,
        component_flow_df=component_flow_df,
        balance_check_df=balance_check_df,
        reaction_summary_df=reaction_summary_df,
        species_conversion_df=species_conversion_df,
        equation_residuals=eq_residuals,
    )


# ---------------------------------------------------------------------------
# Built-in Demo Presets
# ---------------------------------------------------------------------------

def get_demo_presets() -> Dict[str, MaterialBalanceProblem]:
    p1 = MaterialBalanceProblem(
        species=["A", "B", "C"],
        inlet_streams=[StreamDef("S1", "Feed 1"), StreamDef("S2", "Feed 2")],
        outlet_streams=[StreamDef("S3", "Pure Product"), StreamDef("S4", "Purge")],
        reactions=[[-1.0, -1.0, 1.0]],
        stream_inputs={
            "S1": StreamInput(total_flow=100.0, mode="mole_fraction", values=[0.50, 0.50, 0.00]),
            "S2": StreamInput(total_flow=50.0, mode="mole_fraction", values=[0.00, 1.00, 0.00]),
            "S3": StreamInput(total_flow=None, mode="mole_fraction", values=[0.00, 0.00, 1.00]),
            "S4": StreamInput(total_flow=None, mode="mole_fraction", values=[None, None, 0.00]),
        },
        constraints=[{"type": "conversion", "species": "A", "value": 0.80}],
        flow_unit="kmol/h",
    )

    p2 = MaterialBalanceProblem(
        species=["CH4", "O2", "N2", "CO2", "H2O"],
        inlet_streams=[StreamDef("Fuel", "Natural Gas"), StreamDef("Air", "20% Excess Air")],
        outlet_streams=[StreamDef("Flue", "Flue Gas")],
        reactions=[[-1.0, -2.0, 0.0, 1.0, 2.0]],
        stream_inputs={
            "Fuel": StreamInput(total_flow=100.0, mode="mole_fraction", values=[1.0, 0.0, 0.0, 0.0, 0.0]),
            "Air": StreamInput(total_flow=None, mode="mole_fraction", values=[0.0, 0.21, 0.79, 0.0, 0.0]),
            "Flue": StreamInput(total_flow=None, mode="mole_fraction", values=[None, None, None, None, None]),
        },
        constraints=[
            {"type": "conversion", "species": "CH4", "value": 0.90},
            {"type": "custom", "equation": "n_Air_O2 = 2.4 * n_Fuel_CH4"},
        ],
        flow_unit="mol/min",
    )

    p3 = MaterialBalanceProblem(
        species=["CO", "H2O", "CO2", "H2"],
        inlet_streams=[StreamDef("Feed", "Syngas + Steam")],
        outlet_streams=[StreamDef("Product", "Equilibrium Exit")],
        reactions=[[-1.0, -1.0, 1.0, 1.0]],
        stream_inputs={
            "Feed": StreamInput(total_flow=None, mode="component_flow", values=[50.0, 100.0, 0.0, 0.0]),
            "Product": StreamInput(total_flow=None, mode="mole_fraction", values=[None, None, None, None]),
        },
        constraints=[
            {"type": "equilibrium", "reaction_idx": 1, "stream_id": "Product", "keq": 4.0, "pressure": 1.0}
        ],
        flow_unit="mol/s",
    )

    p4 = MaterialBalanceProblem(
        species=["Benzene", "Toluene", "Xylene"],
        inlet_streams=[StreamDef("Feed", "BTX Mixture")],
        outlet_streams=[StreamDef("Distillate", "Overhead"), StreamDef("Bottoms", "Heavy Cut")],
        reactions=[],
        stream_inputs={
            "Feed": StreamInput(total_flow=100.0, mode="mole_fraction", values=[0.40, 0.40, 0.20]),
            "Distillate": StreamInput(total_flow=None, mode="mole_fraction", values=[None, None, None]),
            "Bottoms": StreamInput(total_flow=None, mode="mole_fraction", values=[None, None, None]),
        },
        constraints=[
            {"type": "split", "basis": "outlet", "species": "Benzene", "stream_id": "Distillate", "fraction": 0.90},
            {"type": "split", "basis": "outlet", "species": "Toluene", "stream_id": "Distillate", "fraction": 0.50},
            {"type": "split", "basis": "outlet", "species": "Xylene", "stream_id": "Distillate", "fraction": 0.05},
        ],
        flow_unit="kmol/h",
    )

    return {
        "1. Synthesis + Purge (A + B → C)": p1,
        "2. Methane Combustion (20% Excess Air)": p2,
        "3. Water-Gas Shift Equilibrium (K_eq = 4)": p3,
        "4. Non-Reacting BTX Splitter (N_r = 0)": p4,
    }


# ---------------------------------------------------------------------------
# Standalone VS Code Verification Suite
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    presets = get_demo_presets()
    for title, prob in presets.items():
        print("=" * 72)
        print(f"TEST: {title}")
        res = solve_material_balance(prob)
        print(f"Status: {res.status} | Max Residual: {res.max_residual:.2e}")
        print(f"Message: {res.message}")
        if res.stream_table_df is not None:
            print("\nStream Table (Mole Fractions):")
            print(res.stream_table_df.to_string(index=False))
            print("\nComponent Flows:")
            print(res.component_flow_df.to_string(index=False))
            print("\nBalance Verification:")
            print(res.balance_check_df.to_string(index=False))
            if not res.reaction_summary_df.empty:
                print("\nReaction Extents:")
                print(res.reaction_summary_df.to_string(index=False))
        assert res.success, f"Benchmark failed: {title} -> {res.message}"
    print("=" * 72)
    print("ALL 4 BENCHMARKS PASSED SUCCESSFULLY.")
    print("To open the web interface, run:  python app.py   (or click Run on app.py)")