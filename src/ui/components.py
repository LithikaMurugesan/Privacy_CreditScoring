import streamlit as st
import pandas as pd


def icon(fa, color="#38bdf8"):
    return f'<i class="fa-solid {fa}" style="color:{color};"></i>'

def icon_header(fa, text, level=2, color="#38bdf8"):
    st.markdown(
        f'<h{level}>{icon(fa, color)} {text}</h{level}>',
        unsafe_allow_html=True
    )

def icon_status(fa, msg, color, bg):
    st.markdown(
        f'<div style="background:{bg}; border-left: 4px solid {color}; '
        f'border-radius: 6px; padding: 0.75rem 1rem; color:{color}; '
        f'font-size: 0.9rem; margin: 0.5rem 0;">'
        f'{icon(fa, color)} {msg}</div>',
        unsafe_allow_html=True,
    )

def render_log_tab(logger):
   
    from src.ui.components import icon_header

    icon_header("fa-terminal", "Full FL Training Log", level=3)

    cdf = logger.client_df()
    gdf = logger.global_df()

    if cdf.empty:
        st.info("No log data yet.")
        return

    st.markdown("**Per-bank metrics per round:**")
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    st.markdown("**Global model metrics per round:**")
    st.dataframe(gdf, use_container_width=True, hide_index=True)

    st.markdown("**Raw log:**")
    all_lines = []
    for rnd in sorted(set(r["round"] for r in logger.records)):
        all_lines.extend(logger.round_lines(rnd))

    st.markdown(
        f'<div class="log-box">{"<br>".join(all_lines)}</div>',
        unsafe_allow_html=True
    )

def render_export_tab(logger):
    import streamlit as st
    from src.ui.components import icon_header

    icon_header("fa-file-csv", "Export Training Data as CSV", level=3)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download client_log.csv",
            logger.client_csv(),
            "client_log.csv",
            "text/csv"
        )

    with c2:
        st.download_button(
            "Download global_log.csv",
            logger.global_csv(),
            "global_log.csv",
            "text/csv"
        )

    with c3:
        combined = logger.combined_csv()
        st.download_button(
            "Download combined.csv",
            combined if combined else b"no data",
            "fl_combined.csv",
            "text/csv"
        )
