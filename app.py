import base64
import html
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from solver import (
    MaterialBalanceProblem,
    SolutionResult,
    StreamDef,
    StreamInput,
    format_reaction_str,
    get_custom_symbol_guide,
    get_demo_presets,
    sanitize_ident,
    solve_material_balance,
)


# ---------------------------------------------------------------------------
# Page Config & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Material Balance Solver",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main-banner {
    background: linear-gradient(135deg, #0f2c59 0%, #1c4e80 100%);
    color: #ffffff;
    padding: 1.1rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 3px 10px rgba(15, 44, 89, 0.18);
}
.main-banner h1 {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
}
.main-banner p {
    margin: 0.2rem 0 0 0;
    font-size: 0.92rem;
    color: #d6e4f0;
}
.card-box {
    background-color: #f8fafc;
    color: #0f2c59;
    border: 1px solid #dbe4ee;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
}
.dof-pill-ok {
    background-color: #e6f4ea;
    border-left: 5px solid #1e8e3e;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    color: #137333;
    font-weight: 500;
}
.dof-pill-warn {
    background-color: #fef7e0;
    border-left: 5px solid #f9ab00;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    color: #8a5300;
    font-weight: 500;
}
.dof-pill-err {
    background-color: #fce8e6;
    border-left: 5px solid #d93025;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    color: #a50e0e;
    font-weight: 500;
}
.rxn-preview {
    background-color: #eef4fb;
    border: 1px solid #bfd7f2;
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    font-family: monospace;
    font-size: 0.98rem;
    color: #0f2c59;
    margin-top: 0.35rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Initialization & Resizing Helpers
# ---------------------------------------------------------------------------

def default_species_name(idx: int) -> str:
    if idx < 26:
        return chr(ord("A") + idx)
    return f"Sp{idx + 1}"


def init_state() -> None:
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.view = "Input"
    st.session_state.step = 1
    st.session_state.form_version = 0
    st.session_state.flow_unit = "kmol/h"
    st.session_state.species = ["A", "B", "C"]
    st.session_state.inlet_streams = [
        {"id": "S1", "desc": "Feed 1"},
        {"id": "S2", "desc": "Feed 2"},
    ]
    st.session_state.outlet_streams = [
        {"id": "S3", "desc": "Product"},
        {"id": "S4", "desc": "Purge / Byproduct"},
    ]
    st.session_state.reactions = [[-1.0, -1.0, 1.0]]
    st.session_state.stream_inputs = {
        "S1": {"total_flow": None, "mode": "mole_fraction", "values": [None, None, None]},
        "S2": {"total_flow": None, "mode": "mole_fraction", "values": [None, None, None]},
        "S3": {"total_flow": None, "mode": "mole_fraction", "values": [None, None, None]},
        "S4": {"total_flow": None, "mode": "mole_fraction", "values": [None, None, None]},
    }
    st.session_state.constraints = []
    st.session_state.last_result = None


def load_problem_into_state(prob: MaterialBalanceProblem) -> None:
    st.session_state.flow_unit = prob.flow_unit
    st.session_state.species = list(prob.species)
    st.session_state.inlet_streams = [{"id": s.id, "desc": s.desc} for s in prob.inlet_streams]
    st.session_state.outlet_streams = [{"id": s.id, "desc": s.desc} for s in prob.outlet_streams]
    st.session_state.reactions = [list(r) for r in prob.reactions]

    si_dict: Dict[str, Dict[str, Any]] = {}
    for st_def in prob.all_streams:
        inp = prob.stream_inputs.get(st_def.id)
        if inp:
            si_dict[st_def.id] = {
                "total_flow": inp.total_flow,
                "mode": inp.mode,
                "values": list(inp.values),
            }
        else:
            si_dict[st_def.id] = {
                "total_flow": None,
                "mode": "mole_fraction",
                "values": [None] * prob.n_c,
            }
    st.session_state.stream_inputs = si_dict
    st.session_state.constraints = [dict(c) for c in prob.constraints]
    st.session_state.last_result = None
    st.session_state.form_version += 1


def reset_blank_state() -> None:
    blank = MaterialBalanceProblem(
        species=["A", "B", "C"],
        inlet_streams=[StreamDef("S1", "Feed 1")],
        outlet_streams=[StreamDef("S2", "Product")],
        reactions=[[-1.0, -1.0, 1.0]],
        stream_inputs={
            "S1": StreamInput(None, "mole_fraction", [None, None, None]),
            "S2": StreamInput(None, "mole_fraction", [None, None, None]),
        },
        constraints=[],
        flow_unit="kmol/h",
    )
    load_problem_into_state(blank)
    st.session_state.step = 1
    st.session_state.view = "Input"


def sync_dimensions(new_nc: int, new_nin: int, new_nout: int, new_nr: int) -> None:
    # 1. Resize species
    species: List[str] = st.session_state.species
    if new_nc < len(species):
        species = species[:new_nc]
    elif new_nc > len(species):
        idx = len(species)
        while len(species) < new_nc:
            cand = default_species_name(idx)
            while cand in species:
                idx += 1
                cand = default_species_name(idx)
            species.append(cand)
            idx += 1
    st.session_state.species = species

    # 2. Resize reactions
    reactions: List[List[float]] = st.session_state.reactions
    resized_rxns: List[List[float]] = []
    for r in reactions[:new_nr]:
        if len(r) >= new_nc:
            resized_rxns.append(r[:new_nc])
        else:
            resized_rxns.append(r + [0.0] * (new_nc - len(r)))
    while len(resized_rxns) < new_nr:
        resized_rxns.append([0.0] * new_nc)
    st.session_state.reactions = resized_rxns

    # 3. Resize inlet & outlet streams uniquely
    inlets: List[Dict[str, str]] = st.session_state.inlet_streams
    outlets: List[Dict[str, str]] = st.session_state.outlet_streams

    def next_stream_id(used: set) -> str:
        n = 1
        while f"S{n}" in used:
            n += 1
        return f"S{n}"

    if new_nin < len(inlets):
        inlets = inlets[:new_nin]
    else:
        used_ids = {s["id"] for s in inlets + outlets}
        while len(inlets) < new_nin:
            sid = next_stream_id(used_ids)
            used_ids.add(sid)
            inlets.append({"id": sid, "desc": f"Feed {len(inlets) + 1}"})

    if new_nout < len(outlets):
        outlets = outlets[:new_nout]
    else:
        used_ids = {s["id"] for s in inlets + outlets}
        while len(outlets) < new_nout:
            sid = next_stream_id(used_ids)
            used_ids.add(sid)
            outlets.append({"id": sid, "desc": f"Outlet {len(outlets) + 1}"})

    st.session_state.inlet_streams = inlets
    st.session_state.outlet_streams = outlets

    # 4. Sync stream_inputs dictionary
    active_ids = [s["id"] for s in inlets + outlets]
    new_inputs: Dict[str, Dict[str, Any]] = {}
    for sid in active_ids:
        old = st.session_state.stream_inputs.get(sid)
        if old:
            vals = list(old["values"])
            if len(vals) >= new_nc:
                vals = vals[:new_nc]
            else:
                vals = vals + [None] * (new_nc - len(vals))
            new_inputs[sid] = {
                "total_flow": old["total_flow"],
                "mode": old["mode"],
                "values": vals,
            }
        else:
            new_inputs[sid] = {
                "total_flow": None,
                "mode": "mole_fraction",
                "values": [None] * new_nc,
            }
    st.session_state.stream_inputs = new_inputs

    # 5. Filter out orphaned constraints
    valid_species = set(species)
    valid_streams = set(active_ids)
    valid_outlets = {s["id"] for s in outlets}
    cleaned_constraints = []
    for c in st.session_state.constraints:
        ct = c.get("type")
        if ct == "conversion" and c.get("species") not in valid_species:
            continue
        if ct == "extent" and int(c.get("reaction_idx", 1)) > new_nr:
            continue
        if ct == "split" and (c.get("species") not in valid_species or c.get("stream_id") not in valid_outlets):
            continue
        if ct == "flow_ratio" and (c.get("num_stream") not in valid_streams or c.get("den_stream") not in valid_streams):
            continue
        if ct == "molar_ratio" and (
            c.get("stream_id") not in valid_streams
            or c.get("num_species") not in valid_species
            or c.get("den_species") not in valid_species
        ):
            continue
        if ct == "equilibrium" and (
            int(c.get("reaction_idx", 1)) > new_nr or c.get("stream_id") not in valid_outlets
        ):
            continue
        cleaned_constraints.append(c)
    st.session_state.constraints = cleaned_constraints


def build_problem_from_state() -> MaterialBalanceProblem:
    stream_inputs = {
        sid: StreamInput(
            total_flow=data["total_flow"],
            mode=data["mode"],
            values=list(data["values"]),
        )
        for sid, data in st.session_state.stream_inputs.items()
    }
    return MaterialBalanceProblem(
        species=list(st.session_state.species),
        inlet_streams=[StreamDef(s["id"], s["desc"]) for s in st.session_state.inlet_streams],
        outlet_streams=[StreamDef(s["id"], s["desc"]) for s in st.session_state.outlet_streams],
        reactions=[list(r) for r in st.session_state.reactions],
        stream_inputs=stream_inputs,
        constraints=[dict(c) for c in st.session_state.constraints],
        flow_unit=st.session_state.flow_unit,
    )


def parse_optional_float(raw: str) -> Tuple[Optional[float], Optional[str]]:
    txt = raw.strip()
    if txt == "":
        return None, None
    try:
        if "/" in txt:
            parts = txt.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1]), None
        return float(txt), None
    except Exception:
        return None, f"Invalid number '{txt}'"


def fmt_num(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val:g}"


# ---------------------------------------------------------------------------
# Dynamic SVG Reactor Diagram
# ---------------------------------------------------------------------------

def render_reactor_svg(
    problem: MaterialBalanceProblem,
    solution: Optional[SolutionResult] = None,
) -> str:
    n_in = problem.n_in
    n_out = problem.n_out
    max_ports = max(n_in, n_out, 2)
    height = max(210, 60 + max_ports * 46)
    width = 620
    box_x, box_w = 195, 230
    box_h = max(120, height - 60)
    box_y = (height - box_h) / 2

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="font-family: Arial, sans-serif; background:#ffffff; border-radius:8px; border:1px solid #dbe4ee;">',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 1 L 10 5 L 0 9 z" fill="#1c4e80" />',
        '</marker>',
        '</defs>',
        # Reactor block
        f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="12" ry="12" '
        f'fill="#eef4fb" stroke="#1c4e80" stroke-width="2.5"/>',
        f'<text x="{box_x + box_w / 2}" y="{box_y + 30}" text-anchor="middle" '
        f'font-size="16" font-weight="bold" fill="#0f2c59">REACTOR / UNIT</text>',
        f'<text x="{box_x + box_w / 2}" y="{box_y + 52}" text-anchor="middle" '
        f'font-size="12" fill="#3b5b84">{problem.n_c} Species | {problem.n_r} Reaction(s)</text>',
    ]

    # Show up to 3 reactions inside the box
    for m in range(min(problem.n_r, 3)):
        rxn_txt = html.escape(format_reaction_str(problem.species, problem.reactions[m]))
        if len(rxn_txt) > 30:
            rxn_txt = rxn_txt[:27] + "..."
        lines.append(
            f'<text x="{box_x + box_w / 2}" y="{box_y + 78 + m * 18}" text-anchor="middle" '
            f'font-size="11" fill="#1c4e80">{rxn_txt}</text>'
        )
    if problem.n_r > 3:
        lines.append(
            f'<text x="{box_x + box_w / 2}" y="{box_y + 78 + 3 * 18}" text-anchor="middle" '
            f'font-size="11" fill="#5c7699">+{problem.n_r - 3} more reaction(s)</text>'
        )
    elif problem.n_r == 0:
        lines.append(
            f'<text x="{box_x + box_w / 2}" y="{box_y + 82}" text-anchor="middle" '
            f'font-size="11" fill="#5c7699">(Non-reactive system)</text>'
        )

    # Inlet arrows
    for i, st_def in enumerate(problem.inlet_streams):
        y = box_y + (i + 1) * box_h / (n_in + 1)
        lines.append(
            f'<line x1="25" y1="{y}" x2="{box_x - 4}" y2="{y}" stroke="#1c4e80" '
            f'stroke-width="2.2" marker-end="url(#arrow)"/>'
        )
        lbl = html.escape(st_def.id)
        if solution and solution.total_flows is not None:
            f_val = solution.total_flows[i]
            lbl += f" ({f_val:.2f} {html.escape(problem.flow_unit)})"
        elif st_def.desc:
            desc_short = html.escape(st_def.desc[:16])
            lbl += f" ({desc_short})"
        lines.append(
            f'<text x="108" y="{y - 7}" text-anchor="middle" font-size="12" '
            f'font-weight="600" fill="#0f2c59">{lbl}</text>'
        )

    # Outlet arrows
    for j, st_def in enumerate(problem.outlet_streams):
        y = box_y + (j + 1) * box_h / (n_out + 1)
        lines.append(
            f'<line x1="{box_x + box_w}" y1="{y}" x2="{width - 25}" y2="{y}" '
            f'stroke="#1c4e80" stroke-width="2.2" marker-end="url(#arrow)"/>'
        )
        lbl = html.escape(st_def.id)
        if solution and solution.total_flows is not None:
            f_val = solution.total_flows[n_in + j]
            lbl += f" ({f_val:.2f} {html.escape(problem.flow_unit)})"
        elif st_def.desc:
            desc_short = html.escape(st_def.desc[:16])
            lbl += f" ({desc_short})"
        lines.append(
            f'<text x="{box_x + box_w + 95}" y="{y - 7}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#0f2c59">{lbl}</text>'
        )

    lines.append("</svg>")
    svg_bytes = "\n".join(lines).encode("utf-8")
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f'<div style="text-align:center;"><img src="data:image/svg+xml;base64,{b64}" style="max-width:100%; height:auto;"/></div>'


# ---------------------------------------------------------------------------
# App Initialization & Sidebar
# ---------------------------------------------------------------------------

init_state()

st.markdown(
    """
    <div class="main-banner">
        <div>
            <h1>⚗️ Material Balance Solver</h1>
            <p>Generalized Steady-State Reactor &amp; Process Material Balance Interface</p>
        </div>
        <div style="text-align:right; font-size:0.85rem; color:#d6e4f0;">
            Variables = (N<sub>in</sub> + N<sub>out</sub>)&middot;N<sub>c</sub> + N<sub>r</sub><br/>
            Balances = N<sub>c</sub>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Navigation")
    nav_options = ["Input", "Equations", "Results", "Help"]
    nav_labels = {
        "Input": "📝 Input Wizard",
        "Equations": "ƒx Equations & DOF",
        "Results": "📊 Results & Tables",
        "Help": "❓ Guide & Syntax",
    }
    for opt in nav_options:
        btn_type = "primary" if st.session_state.view == opt else "secondary"
        if st.button(nav_labels[opt], use_container_width=True, type=btn_type, key=f"nav_{opt}"):
            st.session_state.view = opt
            st.rerun()

    st.divider()
    st.subheader("Quick Demo Presets")
    presets = get_demo_presets()
    selected_preset = st.selectbox("Choose example", list(presets.keys()), label_visibility="collapsed")
    if st.button("Load & Solve Preset", use_container_width=True):
        load_problem_into_state(presets[selected_preset])
        prob_loaded = build_problem_from_state()
        st.session_state.last_result = solve_material_balance(prob_loaded)
        st.session_state.step = 5
        st.session_state.view = "Results"
        st.rerun()

    st.divider()
    st.subheader("Settings")
    units = ["kmol/h", "mol/min", "mol/s", "mol/h", "lbmol/h"]
    unit_idx = units.index(st.session_state.flow_unit) if st.session_state.flow_unit in units else 0
    st.session_state.flow_unit = st.selectbox("Molar Flow Unit", units, index=unit_idx)

    if st.button("↺ Reset Blank System", use_container_width=True):
        reset_blank_state()
        st.rerun()


# ---------------------------------------------------------------------------
# View: Input Wizard (Steps 1 to 5)
# ---------------------------------------------------------------------------

if st.session_state.view == "Input":
    step_titles = [
        (1, "System"),
        (2, "Species"),
        (3, "Streams"),
        (4, "Reactions"),
        (5, "Data Entry"),
    ]
    cols = st.columns(5)
    for col, (s_num, s_lbl) in zip(cols, step_titles):
        active = st.session_state.step == s_num
        done = st.session_state.step > s_num
        prefix = "● " if active else ("✓ " if done else f"{s_num}. ")
        label = f"{prefix}{s_lbl}" if not active else f"{s_num}. {s_lbl}"
        if col.button(label, use_container_width=True, type="primary" if active else "secondary", key=f"topstep_{s_num}"):
            st.session_state.step = s_num
            st.rerun()

    st.markdown("---")
    fv = st.session_state.form_version

    # -------------------- STEP 1: SYSTEM SPECIFICATION --------------------
    if st.session_state.step == 1:
        st.subheader("1. System Specification")
        st.caption("Define the size of your chemical system to dynamically generate the solver matrices.")

        c_left, c_right = st.columns([1.05, 1.25], gap="large")
        with c_left:
            st.markdown('<div class="card-box"><strong>System Details</strong></div>', unsafe_allow_html=True)
            new_nc = st.number_input(
                "Number of species (N_c)",
                min_value=1,
                max_value=15,
                value=len(st.session_state.species),
                step=1,
                key=f"nc_input_v{fv}",
            )
            new_nin = st.number_input(
                "Number of inlet streams (N_in)",
                min_value=1,
                max_value=10,
                value=len(st.session_state.inlet_streams),
                step=1,
                key=f"nin_input_v{fv}",
            )
            new_nout = st.number_input(
                "Number of outlet streams (N_out)",
                min_value=1,
                max_value=10,
                value=len(st.session_state.outlet_streams),
                step=1,
                key=f"nout_input_v{fv}",
            )
            new_nr = st.number_input(
                "Number of reactions (N_r)",
                min_value=0,
                max_value=10,
                value=len(st.session_state.reactions),
                step=1,
                key=f"nr_input_v{fv}",
            )

            if (
                new_nc != len(st.session_state.species)
                or new_nin != len(st.session_state.inlet_streams)
                or new_nout != len(st.session_state.outlet_streams)
                or new_nr != len(st.session_state.reactions)
            ):
                sync_dimensions(int(new_nc), int(new_nin), int(new_nout), int(new_nr))
                st.session_state.last_result = None
                st.rerun()

            prob_preview = build_problem_from_state()
            req_specs = prob_preview.n_vars - prob_preview.n_c
            st.info(
                f"**System Accounting:**\n"
                f"- Total unknowns = `({prob_preview.n_in} + {prob_preview.n_out}) × {prob_preview.n_c} + {prob_preview.n_r}` = **{prob_preview.n_vars}**\n"
                f"- Independent species material balances = **{prob_preview.n_c}**\n"
                f"- Independent specifications needed from you = **{req_specs}**"
            )

        with c_right:
            st.markdown('<div class="card-box"><strong>System Overview Diagram</strong></div>', unsafe_allow_html=True)
            st.markdown(render_reactor_svg(build_problem_from_state()), unsafe_allow_html=True)
            st.markdown(
                """
                **Workflow:**
                1. Set system dimensions ($N_c, N_{in}, N_{out}, N_r$)
                2. Name chemical species
                3. Define inlet and outlet stream IDs
                4. Enter stoichiometric coefficients ($\\nu$)
                5. Enter known stream data & constraints, then **Solve**
                """
            )

    # -------------------- STEP 2: SPECIES DEFINITION --------------------
    elif st.session_state.step == 2:
        st.subheader("2. Species Definition")
        st.caption(f"Enter unique names for all {len(st.session_state.species)} chemical species.")

        c_left, c_right = st.columns([1.4, 1.0], gap="large")
        with c_left:
            updated_species = []
            for idx, sp_name in enumerate(st.session_state.species):
                r_col1, r_col2 = st.columns([0.15, 0.85])
                r_col1.markdown(f"**#{idx + 1}**")
                new_name = r_col2.text_input(
                    f"Species {idx + 1}",
                    value=sp_name,
                    key=f"sp_name_{idx}_v{fv}",
                    label_visibility="collapsed",
                ).strip()
                updated_species.append(new_name)

            # Propagate renames to constraints cleanly
            for old_name, new_name in zip(st.session_state.species, updated_species):
                if old_name != new_name and new_name:
                    for c in st.session_state.constraints:
                        for field_k in ("species", "num_species", "den_species"):
                            if c.get(field_k) == old_name:
                                c[field_k] = new_name
            st.session_state.species = updated_species

            if any(not s for s in updated_species):
                st.error("Species names cannot be blank.")
            elif len(set(updated_species)) != len(updated_species):
                st.error("Species names must be unique.")

        with c_right:
            st.info(
                "**Guidelines:**\n"
                "- Use consistent, unique identifiers (e.g., `A`, `B`, `CH4`, `O2`, `CO2`, `H2O`).\n"
                "- Avoid spaces or mathematical symbols (`+`, `-`, `*`, `/`) if you plan to reference species in custom equations."
            )

    # -------------------- STEP 3: STREAM DEFINITION --------------------
    elif st.session_state.step == 3:
        st.subheader("3. Stream Definition")
        st.caption("Name your inlet and outlet streams and add optional descriptions.")

        c_in, c_out = st.columns(2, gap="large")

        def render_stream_editor(col, streams_list: List[Dict[str, str]], title: str, tag: str) -> List[Dict[str, str]]:
            updated = []
            with col:
                st.markdown(f"**{title} ({len(streams_list)})**")
                h1, h2 = st.columns([0.4, 0.6])
                h1.caption("Stream ID")
                h2.caption("Description (optional)")
                for i, s_obj in enumerate(streams_list):
                    c1, c2 = st.columns([0.4, 0.6])
                    new_id = c1.text_input(
                        f"{tag}_id_{i}",
                        value=s_obj["id"],
                        key=f"st_id_{tag}_{i}_v{fv}",
                        label_visibility="collapsed",
                    ).strip()
                    new_desc = c2.text_input(
                        f"{tag}_desc_{i}",
                        value=s_obj["desc"],
                        key=f"st_desc_{tag}_{i}_v{fv}",
                        label_visibility="collapsed",
                    )
                    updated.append({"id": new_id, "desc": new_desc})
            return updated

        old_in = list(st.session_state.inlet_streams)
        old_out = list(st.session_state.outlet_streams)
        new_in = render_stream_editor(c_in, old_in, "Input Streams", "in")
        new_out = render_stream_editor(c_out, old_out, "Output Streams", "out")

        # Migrate stream_inputs & constraints when stream IDs are edited
        all_old = [s["id"] for s in old_in + old_out]
        all_new = [s["id"] for s in new_in + new_out]
        if len(set(all_new)) == len(all_new) and all(bool(sid) for sid in all_new):
            if all_old != all_new:
                new_si = {}
                for old_id, new_id in zip(all_old, all_new):
                    new_si[new_id] = st.session_state.stream_inputs.get(
                        old_id,
                        {"total_flow": None, "mode": "mole_fraction", "values": [None] * len(st.session_state.species)},
                    )
                    if old_id != new_id:
                        for c in st.session_state.constraints:
                            for fk in ("stream_id", "num_stream", "den_stream"):
                                if c.get(fk) == old_id:
                                    c[fk] = new_id
                st.session_state.stream_inputs = new_si
            st.session_state.inlet_streams = new_in
            st.session_state.outlet_streams = new_out
        else:
            st.error("All stream IDs must be non-empty and unique across inlets and outlets.")

    # -------------------- STEP 4: REACTION SPECIFICATION --------------------
    elif st.session_state.step == 4:
        st.subheader("4. Reaction Specification")
        if len(st.session_state.reactions) == 0:
            st.info(
                "No chemical reactions configured (`N_r = 0`). The solver will treat this unit as a "
                "non-reactive mixer, splitter, or separator. Proceed to Step 5."
            )
        else:
            st.caption("Enter stoichiometric coefficients (negative for reactants, positive for products, 0 for inerts).")
            species = st.session_state.species
            for m, rxn_coeffs in enumerate(st.session_state.reactions):
                st.markdown(f"**Reaction {m + 1}**")
                scols = st.columns(len(species))
                new_coeffs = []
                for k, sp_name in enumerate(species):
                    val = scols[k].number_input(
                        sp_name,
                        value=float(rxn_coeffs[k]),
                        step=1.0,
                        format="%.4g",
                        key=f"nu_{m}_{k}_v{fv}",
                    )
                    new_coeffs.append(float(val))
                st.session_state.reactions[m] = new_coeffs
                st.markdown(
                    f'<div class="rxn-preview">Reaction {m + 1}: {html.escape(format_reaction_str(species, new_coeffs))}</div>',
                    unsafe_allow_html=True,
                )
                st.write("")

    # -------------------- STEP 5: DATA ENTRY & SOLVE --------------------
    elif st.session_state.step == 5:
        st.subheader("5. Data Entry & Constraints")
        st.caption("Leave unknown cells blank (`—`). Enter known flows, compositions, conversions, splits, or equilibrium conditions.")

        species = st.session_state.species
        all_streams = st.session_state.inlet_streams + st.session_state.outlet_streams
        inlet_ids = {s["id"] for s in st.session_state.inlet_streams}
        outlet_ids = [s["id"] for s in st.session_state.outlet_streams]
        parse_errors: List[str] = []

        col_data, col_const = st.columns([1.45, 1.0], gap="large")

        with col_data:
            st.markdown(f"**Stream Data Matrix (`{st.session_state.flow_unit}`)**")
            # Header row
            hdr_widths = [0.75, 1.15, 0.9] + [0.75] * len(species)
            hcols = st.columns(hdr_widths)
            hcols[0].markdown("**Stream**")
            hcols[1].markdown("**Entry Mode**")
            hcols[2].markdown("**Total Flow (F)**")
            for k, sp_name in enumerate(species):
                hcols[3 + k].markdown(f"**{sp_name}**")

            for s_obj in all_streams:
                sid = s_obj["id"]
                s_data = st.session_state.stream_inputs[sid]
                rcols = st.columns(hdr_widths)
                tag = "In" if sid in inlet_ids else "Out"
                rcols[0].markdown(f"`{sid}` ({tag})")

                mode_opts = ["mole_fraction", "component_flow"]
                mode_labels = {"mole_fraction": "Mole Frac (x)", "component_flow": "Comp Flow (n)"}
                cur_mode = s_data["mode"]
                sel_mode = rcols[1].selectbox(
                    f"mode_{sid}",
                    options=mode_opts,
                    index=mode_opts.index(cur_mode),
                    format_func=lambda m: mode_labels[m],
                    key=f"mode_{sid}_v{fv}",
                    label_visibility="collapsed",
                )
                s_data["mode"] = sel_mode

                raw_f = rcols[2].text_input(
                    f"F_{sid}",
                    value=fmt_num(s_data["total_flow"]),
                    placeholder="—",
                    key=f"F_{sid}_v{fv}",
                    label_visibility="collapsed",
                )
                f_val, f_err = parse_optional_float(raw_f)
                if f_err:
                    parse_errors.append(f"Stream {sid} Total Flow: {f_err}")
                s_data["total_flow"] = f_val

                new_vals: List[Optional[float]] = []
                for k, sp_name in enumerate(species):
                    raw_v = rcols[3 + k].text_input(
                        f"v_{sid}_{k}",
                        value=fmt_num(s_data["values"][k]),
                        placeholder="—",
                        key=f"val_{sid}_{k}_v{fv}",
                        label_visibility="collapsed",
                    )
                    v_val, v_err = parse_optional_float(raw_v)
                    if v_err:
                        parse_errors.append(f"Stream {sid} {sp_name}: {v_err}")
                    new_vals.append(v_val)
                s_data["values"] = new_vals

            if parse_errors:
                for pe in parse_errors:
                    st.error(pe)

        with col_const:
            st.markdown("**Additional Constraints**")

            with st.expander("➕ Add Conversion / Extent / Split / Equilibrium", expanded=True):
                c_types = [
                    "Conversion (X)",
                    "Outlet Split / Recovery",
                    "Stream Flow Ratio",
                    "In-Stream Molar Ratio",
                    "Reaction Extent (ξ)",
                    "Equilibrium (K_eq)",
                    "Custom Equation",
                ]
                chosen_type = st.selectbox("Constraint Type", c_types, key="add_ctype")

                if chosen_type == "Conversion (X)":
                    c1, c2 = st.columns(2)
                    sp_sel = c1.selectbox("Species", species, key="conv_sp")
                    x_val = c2.number_input("Fractional Conversion (0–1)", 0.0, 1.0, 0.80, step=0.05, key="conv_val")
                    if st.button("Add Conversion", use_container_width=True):
                        st.session_state.constraints.append({"type": "conversion", "species": sp_sel, "value": float(x_val)})
                        st.rerun()

                elif chosen_type == "Outlet Split / Recovery":
                    c1, c2 = st.columns(2)
                    sp_sel = c1.selectbox("Species", species, key="spl_sp")
                    st_sel = c2.selectbox("Outlet Stream", outlet_ids, key="spl_st")
                    c3, c4 = st.columns(2)
                    basis_sel = c3.selectbox(
                        "Relative To",
                        ["outlet", "feed"],
                        format_func=lambda b: "Total Outlet" if b == "outlet" else "Total Feed",
                        key="spl_basis",
                    )
                    frac_val = c4.number_input("Fraction (0–1)", 0.0, 1.0, 0.90, step=0.05, key="spl_frac")
                    if st.button("Add Split Constraint", use_container_width=True):
                        st.session_state.constraints.append(
                            {"type": "split", "basis": basis_sel, "species": sp_sel, "stream_id": st_sel, "fraction": float(frac_val)}
                        )
                        st.rerun()

                elif chosen_type == "Stream Flow Ratio":
                    all_ids = [s["id"] for s in all_streams]
                    if len(all_ids) < 2:
                        st.info("Need at least 2 streams.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        s1 = c1.selectbox("Stream A", all_ids, index=0, key="fr_s1")
                        s2 = c2.selectbox("Stream B", all_ids, index=1, key="fr_s2")
                        rat = c3.number_input("Ratio (F_A / F_B)", min_value=0.0, value=1.0, step=0.1, key="fr_rat")
                        if st.button("Add Flow Ratio", use_container_width=True):
                            st.session_state.constraints.append(
                                {"type": "flow_ratio", "num_stream": s1, "den_stream": s2, "ratio": float(rat)}
                            )
                            st.rerun()

                elif chosen_type == "In-Stream Molar Ratio":
                    if len(species) < 2:
                        st.info("Need at least 2 species.")
                    else:
                        all_ids = [s["id"] for s in all_streams]
                        st_sel = st.selectbox("Stream", all_ids, key="mr_st")
                        c1, c2, c3 = st.columns(3)
                        sp1 = c1.selectbox("Species 1", species, index=0, key="mr_sp1")
                        sp2 = c2.selectbox("Species 2", species, index=1, key="mr_sp2")
                        rat = c3.number_input("Ratio (n_1 / n_2)", min_value=0.0, value=1.0, step=0.1, key="mr_rat")
                        if st.button("Add Molar Ratio", use_container_width=True):
                            st.session_state.constraints.append(
                                {"type": "molar_ratio", "stream_id": st_sel, "num_species": sp1, "den_species": sp2, "ratio": float(rat)}
                            )
                            st.rerun()

                elif chosen_type == "Reaction Extent (ξ)":
                    if len(st.session_state.reactions) == 0:
                        st.info("No reactions defined.")
                    else:
                        c1, c2 = st.columns(2)
                        rxn_i = c1.selectbox("Reaction #", list(range(1, len(st.session_state.reactions) + 1)), key="ext_rxn")
                        ext_v = c2.number_input(f"Extent ({st.session_state.flow_unit})", value=10.0, step=1.0, key="ext_val")
                        if st.button("Add Extent", use_container_width=True):
                            st.session_state.constraints.append({"type": "extent", "reaction_idx": int(rxn_i), "value": float(ext_v)})
                            st.rerun()

                elif chosen_type == "Equilibrium (K_eq)":
                    if len(st.session_state.reactions) == 0:
                        st.info("No reactions defined.")
                    else:
                        c1, c2 = st.columns(2)
                        rxn_i = c1.selectbox("Reaction #", list(range(1, len(st.session_state.reactions) + 1)), key="eq_rxn")
                        st_sel = c2.selectbox("Equilibrium Stream", outlet_ids, key="eq_st")
                        c3, c4 = st.columns(2)
                        keq_v = c3.number_input("K_eq", min_value=1e-9, value=4.0, format="%.4g", key="eq_keq")
                        p_v = c4.number_input("Pressure P (atm/bar)", min_value=1e-6, value=1.0, format="%.4g", key="eq_p")
                        if st.button("Add Equilibrium", use_container_width=True):
                            st.session_state.constraints.append(
                                {"type": "equilibrium", "reaction_idx": int(rxn_i), "stream_id": st_sel, "keq": float(keq_v), "pressure": float(p_v)}
                            )
                            st.rerun()

                elif chosen_type == "Custom Equation":
                    eq_txt = st.text_input("Equation (e.g. F_S3 = 2 * F_S4 or n_Air_O2 = 2.4 * n_Fuel_CH4)", key="cust_eq_inp")
                    guide = get_custom_symbol_guide(build_problem_from_state())
                    st.caption(
                        f"Variables: `{', '.join(guide['total_flows'][:4])}`, "
                        f"`{', '.join(guide['component_flows'][:4])}`, "
                        f"`{', '.join(guide['mole_fractions'][:3])}`"
                        + (f", `{', '.join(guide['extents'])}`" if guide["extents"] else "")
                    )
                    if st.button("Add Custom Equation", use_container_width=True) and eq_txt.strip():
                        st.session_state.constraints.append({"type": "custom", "equation": eq_txt.strip()})
                        st.rerun()

            # Active constraints list
            if st.session_state.constraints:
                st.markdown("**Active Constraints:**")
                for i, c in enumerate(st.session_state.constraints):
                    cc1, cc2 = st.columns([0.85, 0.15])
                    ct = c["type"]
                    if ct == "conversion":
                        desc = f"Conversion: $X_{{{c['species']}}} = {c['value']:g}$"
                    elif ct == "split":
                        desc = f"Split: `{c['species']}` in `{c['stream_id']}` = {c['fraction']:g} of {c['basis']}"
                    elif ct == "flow_ratio":
                        desc = f"Flow Ratio: `F_{c['num_stream']}` = {c['ratio']:g} × `F_{c['den_stream']}`"
                    elif ct == "molar_ratio":
                        desc = f"Molar Ratio ({c['stream_id']}): `{c['num_species']}` = {c['ratio']:g} × `{c['den_species']}`"
                    elif ct == "extent":
                        desc = f"Extent: $\\xi_{{{c['reaction_idx']}}} = {c['value']:g}$"
                    elif ct == "equilibrium":
                        desc = f"Equilibrium R{c['reaction_idx']} in `{c['stream_id']}` ($K_{{eq}}={c['keq']:g}, P={c['pressure']:g}$)"
                    else:
                        desc = f"Custom: `{c.get('equation', '')}`"
                    cc1.markdown(desc)
                    if cc2.button("✕", key=f"del_c_{i}"):
                        st.session_state.constraints.pop(i)
                        st.rerun()

        # Live DOF Diagnostics Banner & Solve Button
        st.divider()
        live_prob = build_problem_from_state()
        from solver import EquationSystem
        live_dof = EquationSystem(live_prob).analyze_dof()

        if live_dof.status == "INVALID":
            st.markdown(
                f'<div class="dof-pill-err">⚠ Input Error: {" | ".join(live_dof.validation_errors)}</div>',
                unsafe_allow_html=True,
            )
        elif live_dof.status == "UNDER_SPECIFIED":
            st.markdown(
                f'<div class="dof-pill-warn">'
                f'DOF Check: <b>Under-specified (Effective DOF = {live_dof.effective_dof})</b> — '
                f'Variables: {live_dof.n_vars} | Independent Equations: {live_dof.jacobian_rank} '
                f'(need {live_dof.effective_dof} more independent value(s)).'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif live_dof.status == "OVER_SPECIFIED":
            st.markdown(
                f'<div class="dof-pill-warn">'
                f'DOF Check: <b>Over-specified / Redundant</b> — '
                f'Variables: {live_dof.n_vars} | Total Equations: {live_dof.n_total_equations} | Rank: {live_dof.jacobian_rank}. '
                f'Solver will verify consistency.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="dof-pill-ok">'
                f'DOF Check: <b>Well-Posed (DOF = 0)</b> — '
                f'Variables: {live_dof.n_vars} | Independent Equations: {live_dof.jacobian_rank}. Ready to solve!'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        if st.button("▶ Solve Material Balance", type="primary", use_container_width=True, disabled=bool(parse_errors)):
            res = solve_material_balance(live_prob)
            st.session_state.last_result = res
            if res.status in ("SOLVED", "SOLVED_WITH_WARNINGS"):
                st.session_state.view = "Results"
                st.rerun()
            else:
                st.error(res.message)

    # Bottom Prev / Next Step Bar
    st.write("")
    b1, _, b3 = st.columns([1, 3, 1])
    if st.session_state.step > 1:
        if b1.button("← Previous", use_container_width=True):
            st.session_state.step -= 1
            st.rerun()
    if st.session_state.step < 5:
        if b3.button("Next →", type="primary", use_container_width=True):
            st.session_state.step += 1
            st.rerun()


# ---------------------------------------------------------------------------
# View: Equations & DOF
# ---------------------------------------------------------------------------

elif st.session_state.view == "Equations":
    st.subheader("ƒx System Equations & Degrees of Freedom")
    prob = build_problem_from_state()
    from solver import EquationSystem
    sys = EquationSystem(prob)
    dof = sys.analyze_dof()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Variables", dof.n_vars)
    m2.metric("Material Balances (N_c)", dof.n_material_balances)
    m3.metric("User Equations", dof.n_user_specs)
    m4.metric("Effective DOF", dof.effective_dof)

    if dof.validation_errors:
        for err in dof.validation_errors:
            st.error(err)
    for w in dof.warnings:
        st.warning(w)

    st.markdown("### Species Material Balance Form")
    st.latex(r"\sum_{i \in \text{in}} n_{i,k} - \sum_{j \in \text{out}} n_{j,k} + \sum_{m=1}^{N_r} \nu_{k,m}\,\xi_m = 0 \quad (k = 1 \dots N_c)")

    st.markdown("### Active Equations in Current System")
    for idx, lbl in enumerate(dof.equation_labels, start=1):
        st.markdown(f"`{idx:02d}.` {lbl}")


# ---------------------------------------------------------------------------
# View: Results & Stream Tables
# ---------------------------------------------------------------------------

elif st.session_state.view == "Results":
    st.subheader("📊 Results — Material Balance Solution")
    res: Optional[SolutionResult] = st.session_state.last_result
    prob = build_problem_from_state()

    if res is None:
        st.info("No solved results yet. Complete the **Input Wizard** and click **Solve Material Balance**.")
    elif res.status in ("UNDER_SPECIFIED", "CONTRADICTORY", "FAILED"):
        st.error(res.message)
        if res.equation_residuals:
            with st.expander("Inspect Equation Residuals"):
                for lbl, rval in res.equation_residuals:
                    st.write(f"`{lbl}` → residual = `{rval:.4e}`")
    else:
        if res.status == "SOLVED":
            st.success(f"✓ {res.message} (Max residual: {res.max_residual:.2e})")
        else:
            st.warning(f"⚠ {res.message} (Max residual: {res.max_residual:.2e})")
            for w in res.dof.warnings:
                st.caption(f"Note: {w}")

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Species (N_c)", prob.n_c)
        k2.metric("Input Streams (N_in)", prob.n_in)
        k3.metric("Output Streams (N_out)", prob.n_out)
        k4.metric("Reactions (N_r)", prob.n_r)

        # Solved Flowsheet Diagram
        st.markdown(render_reactor_svg(prob, res), unsafe_allow_html=True)
        st.write("")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Stream Table (Mole Fractions)", "Component Molar Flows", "Balance Verification", "Reactions & Conversions"]
        )
        with tab1:
            st.dataframe(res.stream_table_df, use_container_width=True, hide_index=True)
            csv_mf = res.stream_table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Export Stream Table (CSV)",
                data=csv_mf,
                file_name="material_balance_streams.csv",
                mime="text/csv",
            )
        with tab2:
            st.dataframe(res.component_flow_df, use_container_width=True, hide_index=True)
            csv_cf = res.component_flow_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Export Component Flows (CSV)",
                data=csv_cf,
                file_name="material_balance_component_flows.csv",
                mime="text/csv",
            )
        with tab3:
            st.dataframe(res.balance_check_df, use_container_width=True, hide_index=True)
        with tab4:
            if res.reaction_summary_df is not None and not res.reaction_summary_df.empty:
                st.markdown("**Reaction Extents**")
                st.dataframe(res.reaction_summary_df, use_container_width=True, hide_index=True)
            st.markdown("**Species Overall Conversions**")
            st.dataframe(res.species_conversion_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# View: Help & Syntax
# ---------------------------------------------------------------------------

elif st.session_state.view == "Help":
    st.subheader("❓ User Guide & Engineering Model")
    st.markdown(
        """
        ### Mathematical Formulation
        This solver treats the reactor/process unit as a **steady-state control volume** with $N_{in}$ inlet streams,
        $N_{out}$ outlet streams, $N_c$ chemical species, and $N_r$ reactions:
        - **Primary Variables:** Component molar flows $n_{s,k}$ for every stream and species, plus reaction extents $\\xi_m$:
          $$N_{\\text{vars}} = (N_{in} + N_{out}) \\cdot N_c + N_r$$
        - **Conservation Equations ($N_c$ equations):**
          $$\\sum_{i \\in \\text{in}} n_{i,k} - \\sum_{j \\in \\text{out}} n_{j,k} + \\sum_{m=1}^{N_r} \\nu_{k,m}\\,\\xi_m = 0$$
        - **Stoichiometric Sign Convention:** Negative ($-$) for reactants, positive ($+$) for products, $0$ for inert species.

        ### Supported Specifications
        1. **Stream Data Matrix:** Enter either `Total Flow (F) + Mole Fractions (x)` or direct `Component Flows (n)`. Leave unknown cells blank.
        2. **Conversion ($X_k$):** Fractional overall conversion $0 \\le X_k \\le 1$ (e.g., `0.80` for 80%).
        3. **Split / Recovery:** Fraction of a species sent to a specific outlet stream (relative to total feed or total outlet).
        4. **Flow & Molar Ratios:** Ratio between two stream total flows, or between two species inside one stream.
        5. **Equilibrium ($K_{eq}$):** Ideal equilibrium expression $\\prod y_k^{\\nu_k} P^{\\Delta\\nu} = K_{eq}$ applied to a chosen outlet stream.
        6. **Custom Equations:** Write algebraic constraints using `F_<Stream>`, `n_<Stream>_<Species>`, `x_<Stream>_<Species>`, and `xi_<RxnNumber>`.
        """
    )