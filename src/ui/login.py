import streamlit as st


def render_login():
    if not st.session_state["authenticated"]:
        # Invisible marker so CSS can detect login page
        st.markdown(
            '<div class="login-page-marker" style="display:none"></div>',
            unsafe_allow_html=True
        )

      
        with st.container(border=True):
           
            st.markdown(
                '''
                <div class="lc-brand">
                    <div class="lc-icon">
                        <i class="fa-solid fa-shield-halved"></i>
                    </div>
                    <h2>Federated Learning</h2>
                    <p>Privacy-Preserving Credit Scoring System</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        
            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    placeholder="Enter username"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter password"
                )

                submitted = st.form_submit_button(
                    "Sign In",
                    use_container_width=True,
                    type="primary"
                )

                if submitted:
                    if username == "admin" and password == "admin123":
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error(
                            "Invalid credentials. Please try again."
                        )

            st.markdown(
                '''
                <div class="lc-demo">
                    <p>
                        Demo Credentials:
                        <strong>admin</strong> /
                        <strong>admin123</strong>
                    </p>
                </div>
                ''',
                unsafe_allow_html=True
            )

        st.stop()
