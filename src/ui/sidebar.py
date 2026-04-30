import streamlit as st
from streamlit_option_menu import option_menu


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '''
            <div style="text-align: center; padding: 1rem 0;
                        border-bottom: 1px solid #334155;">
                <i class="fa-solid fa-building-columns"
                   style="font-size: 2rem; color: #38bdf8;"></i>
                <h3 style="margin: 0.5rem 0 0 0;
                           color: #38bdf8;">
                    FL Credit Scoring
                </h3>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        page = option_menu(
            menu_title=None,
            options=[
                "Overview",
                "Data Explorer",
                "FL Training",
                "Baseline Comparison",
                "Performance Comparison",
                "Privacy Analysis",
                "Credit Predictor"
            ],
            icons=[
                "speedometer2",
                "database",
                "cpu",
                "arrow-left-right",
                "graph-up-arrow",
                "shield-lock-fill",
                "credit-card-2-front"
            ],
            default_index=0,
            styles={
                "container": {
                    "padding": "0",
                    "background-color": "transparent"
                },
                "icon": {
                    "font-size": "16px",
                    "color": "#64748b"
                },
                "nav-link": {
                    "font-size": "0.9rem",
                    "padding": "0.7rem 1rem",
                    "color": "#94a3b8",
                    "border-radius": "8px",
                    "margin": "0.2rem 0",
                },
                "nav-link-selected": {
                    "background-color": "rgba(56,189,248,0.15)",
                    "color": "#38bdf8",
                    "font-weight": "600",
                },
            },
        )

        st.divider()

        username = st.session_state.get("username", "admin")

        st.markdown(
            f'''
            <div style="padding: 0.5rem 0; color: #94a3b8;">
                <i class="fa-solid fa-user"
                   style="margin-right: 0.5rem;"></i>
                {username}
            </div>
            ''',
            unsafe_allow_html=True,
        )

        if st.button(
            "Logout",
            key="logout_btn",
            use_container_width=True
        ):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        return page
